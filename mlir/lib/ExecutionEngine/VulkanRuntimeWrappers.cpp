//===- VulkanRuntimeWrappers.cpp - MLIR Vulkan runner wrapper library -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C runtime wrappers around the VulkanRuntime.
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "VulkanRuntime.h"

// Explicitly export entry points to the vulkan-runtime-wrapper.

#ifdef _WIN32
#define VULKAN_WRAPPER_SYMBOL_EXPORT __declspec(dllexport)
#else
#define VULKAN_WRAPPER_SYMBOL_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

namespace {

class VulkanModule;

// Class to be a thing that can be returned from `mgpuModuleGetFunction`.
struct VulkanFunction {
  VulkanModule *module;
  std::string name;

  VulkanFunction(VulkanModule *module, const char *name)
      : module(module), name(name) {}
};

// Class to own a copy of the SPIR-V provided to `mgpuModuleLoad` and to manage
// allocation of pointers returned from `mgpuModuleGetFunction`.
class VulkanModule {
public:
  VulkanModule(const uint8_t *ptr, size_t sizeInBytes)
      : blob(ptr, ptr + sizeInBytes) {}
  ~VulkanModule() = default;

  VulkanFunction *getFunction(const char *name) {
    return functions.emplace_back(std::make_unique<VulkanFunction>(this, name))
        .get();
  }

  uint8_t *blobData() { return blob.data(); }
  size_t blobSizeInBytes() const { return blob.size(); }

private:
  std::vector<uint8_t> blob;
  std::vector<std::unique_ptr<VulkanFunction>> functions;
};

// Persistent Vulkan device context shared by the host-side memory API
// (mgpuMemAlloc / mgpuMemcpy / mgpuMemFree) and the raw-pointer kernel launch
// path. Buffer device addresses are only meaningful within the VkDevice that
// created the buffer, so device memory allocation and the launch that
// dereferences it must use the same device; this context provides it.
//
// Created lazily on first use and intentionally leaked at process exit. The
// existing per-launch descriptor path (VulkanRuntime) never touches it, so it
// has no effect on memref-based kernels.
class VulkanContext {
public:
  VulkanContext() {
    if (failed(createInstance()) || failed(createDevice())) {
      std::cerr << "failed to create Vulkan context\n";
      abort();
    }
  }

  VkDevice getDevice() const { return device; }
  VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
  VkQueue getQueue() const { return queue; }
  uint32_t getQueueFamilyIndex() const { return queueFamilyIndex; }

  // Allocates a host-visible, device-address-capable buffer of `size` bytes,
  // usable as a storage buffer and transfer source/destination. Returns the
  // buffer device address and outputs the VkBuffer / VkDeviceMemory handles.
  VkDeviceAddress allocate(VkDeviceSize size, VkBuffer &buffer,
                           VkDeviceMemory &memory) {
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer) !=
        VK_SUCCESS)
      return 0;

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, buffer, &memReqs);

    VkMemoryAllocateFlagsInfo allocFlagsInfo = {};
    allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &allocFlagsInfo;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = hostMemoryTypeIndex;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
      return 0;
    if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS)
      return 0;

    VkBufferDeviceAddressInfo addressInfo = {};
    addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addressInfo.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &addressInfo);
  }

  // Copies `size` bytes between host memory and the host-visible device memory
  // (host-visible + coherent, so no explicit flush is required).
  void copyToDevice(VkDeviceMemory memory, const void *src, VkDeviceSize size) {
    void *payload = nullptr;
    if (vkMapMemory(device, memory, 0, size, 0, &payload) != VK_SUCCESS)
      return;
    std::memcpy(payload, src, size);
    vkUnmapMemory(device, memory);
  }

  void copyFromDevice(VkDeviceMemory memory, void *dst, VkDeviceSize size) {
    void *payload = nullptr;
    if (vkMapMemory(device, memory, 0, size, 0, &payload) != VK_SUCCESS)
      return;
    std::memcpy(dst, payload, size);
    vkUnmapMemory(device, memory);
  }

  // Dispatches a compute shader on this context. `pushConstants` (of
  // `pushConstantSize` bytes) becomes the shader's sole push-constant block;
  // for the raw-pointer launch path this carries the kernel's by-value
  // arguments (device addresses and scalars). No descriptor sets are bound, so
  // all memory access goes through PhysicalStorageBuffer device addresses.
  void launchCompute(const uint8_t *shader, uint32_t shaderSize,
                     const char *entryPoint, uint32_t gx, uint32_t gy,
                     uint32_t gz, const void *pushConstants,
                     uint32_t pushConstantSize) {
    VkShaderModuleCreateInfo shaderInfo = {};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = shaderSize;
    shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shader);
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule) !=
        VK_SUCCESS)
      return;

    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = pushConstantSize;

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.pushConstantRangeCount = pushConstantSize ? 1 : 0;
    layoutInfo.pPushConstantRanges = pushConstantSize ? &pushRange : nullptr;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout);

    VkPipelineShaderStageCreateInfo stageInfo = {};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = entryPoint;

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                             &pipeline);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

    VkCommandBufferAllocateInfo cmdInfo = {};
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdInfo.commandPool = commandPool;
    cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &cmdInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (pushConstantSize)
      vkCmdPushConstants(commandBuffer, pipelineLayout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize,
                         pushConstants);
    vkCmdDispatch(commandBuffer, gx, gy, gz);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

private:
  LogicalResult createInstance() {
    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "MLIR Vulkan memory context";
    applicationInfo.pEngineName = "mlir";
    // Buffer device address is core in Vulkan 1.2.
    applicationInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    if (vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS)
      return failure();
    return success();
  }

  LogicalResult createDevice() {
    uint32_t physicalDeviceCount = 0;
    if (vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr) !=
            VK_SUCCESS ||
        physicalDeviceCount == 0)
      return failure();
    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    if (vkEnumeratePhysicalDevices(instance, &physicalDeviceCount,
                                   physicalDevices.data()) != VK_SUCCESS)
      return failure();
    physicalDevice = physicalDevices.front();

    // Find a queue family supporting compute.
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                             queueFamilies.data());
    bool found = false;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        queueFamilyIndex = i;
        found = true;
        break;
      }
    }
    if (!found)
      return failure();

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures = {};
    bdaFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bdaFeatures.bufferDeviceAddress = VK_TRUE;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = &bdaFeatures;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) !=
        VK_SUCCESS)
      return failure();
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    // Find a host-visible, host-coherent memory type for direct mapping.
    VkPhysicalDeviceMemoryProperties memProps = {};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
      VkMemoryPropertyFlags flags = memProps.memoryTypes[i].propertyFlags;
      if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
          (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        hostMemoryTypeIndex = i;
        break;
      }
    }
    return hostMemoryTypeIndex == VK_MAX_MEMORY_TYPES ? failure() : success();
  }

  VkInstance instance{VK_NULL_HANDLE};
  VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
  VkDevice device{VK_NULL_HANDLE};
  uint32_t queueFamilyIndex{0};
  VkQueue queue{VK_NULL_HANDLE};
  uint32_t hostMemoryTypeIndex{VK_MAX_MEMORY_TYPES};
};

// A device allocation tracked by the memory API, keyed by its device address.
struct VulkanAllocation {
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkDeviceSize size;
};

// Lazily-created, intentionally-leaked global context plus the registry of
// live allocations (device address -> allocation).
static VulkanContext &getGlobalVulkanContext() {
  static VulkanContext *context = new VulkanContext();
  return *context;
}

static std::mutex &getAllocationMutex() {
  static std::mutex *mutex = new std::mutex();
  return *mutex;
}

static std::unordered_map<uint64_t, VulkanAllocation> &getAllocationRegistry() {
  static auto *registry = new std::unordered_map<uint64_t, VulkanAllocation>();
  return *registry;
}

class VulkanRuntimeManager {
public:
  VulkanRuntimeManager() = default;
  VulkanRuntimeManager(const VulkanRuntimeManager &) = delete;
  VulkanRuntimeManager operator=(const VulkanRuntimeManager &) = delete;
  ~VulkanRuntimeManager() = default;

  void setResourceData(DescriptorSetIndex setIndex, BindingIndex bindIndex,
                       const VulkanHostMemoryBuffer &memBuffer) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setResourceData(setIndex, bindIndex, memBuffer);
  }

  void setEntryPoint(const char *entryPoint) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setEntryPoint(entryPoint);
  }

  void setNumWorkGroups(NumWorkGroups numWorkGroups) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setNumWorkGroups(numWorkGroups);
  }

  void setShaderModule(uint8_t *shader, uint32_t size) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setShaderModule(shader, size);
  }

  void setMemRefDescriptor(BindingIndex bindIndex,
                           const void *hostRankedDescriptor,
                           uint32_t descriptorByteSize) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setMemRefDescriptor(bindIndex, hostRankedDescriptor,
                                      descriptorByteSize);
  }

  void runOnVulkan() {
    std::lock_guard<std::mutex> lock(mutex);
    if (failed(vulkanRuntime.initRuntime()) || failed(vulkanRuntime.run()) ||
        failed(vulkanRuntime.updateHostMemoryBuffers()) ||
        failed(vulkanRuntime.destroy())) {
      std::cerr << "runOnVulkan failed";
    }
  }

private:
  VulkanRuntime vulkanRuntime;
  std::mutex mutex;
};

} // namespace

template <typename T, int N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {

//===----------------------------------------------------------------------===//
//
// Wrappers intended for mlir-runner. Uses of GPU dialect operations get
// lowered to calls to these functions by GPUToLLVMConversionPass.
//
//===----------------------------------------------------------------------===//

VULKAN_WRAPPER_SYMBOL_EXPORT void *mgpuStreamCreate() {
  return new VulkanRuntimeManager();
}

VULKAN_WRAPPER_SYMBOL_EXPORT void mgpuStreamDestroy(void *vkRuntimeManager) {
  delete static_cast<VulkanRuntimeManager *>(vkRuntimeManager);
}

VULKAN_WRAPPER_SYMBOL_EXPORT void mgpuStreamSynchronize(void *) {
  // Currently a no-op as the other operations are synchronous.
}

//===----------------------------------------------------------------------===//
//
// Host-side device memory API. Allocates host-owned device memory whose buffer
// device address can be passed directly to PhysicalStorageBuffer kernels. This
// mirrors the CUDA/ROCm runner memory API (mgpuMemAlloc / mgpuMemcpy /
// mgpuMemFree) and lets kernels take raw pointers + scalars instead of bound
// memref descriptors.
//
//===----------------------------------------------------------------------===//

// Allocates `sizeBytes` of device memory and returns its buffer device address.
VULKAN_WRAPPER_SYMBOL_EXPORT uint64_t mgpuMemAlloc(uint64_t sizeBytes) {
  VulkanContext &context = getGlobalVulkanContext();
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceAddress address = context.allocate(sizeBytes, buffer, memory);
  if (!address) {
    std::cerr << "mgpuMemAlloc failed\n";
    return 0;
  }
  std::lock_guard<std::mutex> lock(getAllocationMutex());
  getAllocationRegistry()[address] = VulkanAllocation{buffer, memory,
                                                      sizeBytes};
  return address;
}

// Frees device memory previously returned by mgpuMemAlloc.
VULKAN_WRAPPER_SYMBOL_EXPORT void mgpuMemFree(uint64_t address) {
  VulkanContext &context = getGlobalVulkanContext();
  std::lock_guard<std::mutex> lock(getAllocationMutex());
  auto &registry = getAllocationRegistry();
  auto it = registry.find(address);
  if (it == registry.end())
    return;
  vkDestroyBuffer(context.getDevice(), it->second.buffer, nullptr);
  vkFreeMemory(context.getDevice(), it->second.memory, nullptr);
  registry.erase(it);
}

// Copies `sizeBytes` from host memory at `srcHostPtr` into the device
// allocation at device address `dstAddress`. The host pointer is passed as an
// integer so callers can supply a memref's aligned pointer without an
// !llvm.ptr-typed value.
VULKAN_WRAPPER_SYMBOL_EXPORT void mgpuMemcpyHostToDevice(uint64_t dstAddress,
                                                         uint64_t srcHostPtr,
                                                         uint64_t sizeBytes) {
  VulkanContext &context = getGlobalVulkanContext();
  std::lock_guard<std::mutex> lock(getAllocationMutex());
  auto &registry = getAllocationRegistry();
  auto it = registry.find(dstAddress);
  if (it == registry.end()) {
    std::cerr << "mgpuMemcpyHostToDevice: unknown device address\n";
    return;
  }
  context.copyToDevice(it->second.memory,
                       reinterpret_cast<const void *>(srcHostPtr), sizeBytes);
}

// Copies `sizeBytes` from the device allocation at device address `srcAddress`
// into host memory at `dstHostPtr`.
VULKAN_WRAPPER_SYMBOL_EXPORT void mgpuMemcpyDeviceToHost(uint64_t dstHostPtr,
                                                         uint64_t srcAddress,
                                                         uint64_t sizeBytes) {
  VulkanContext &context = getGlobalVulkanContext();
  std::lock_guard<std::mutex> lock(getAllocationMutex());
  auto &registry = getAllocationRegistry();
  auto it = registry.find(srcAddress);
  if (it == registry.end()) {
    std::cerr << "mgpuMemcpyDeviceToHost: unknown device address\n";
    return;
  }
  context.copyFromDevice(it->second.memory,
                         reinterpret_cast<void *>(dstHostPtr), sizeBytes);
}

// C-interface forwarders. The test Vulkan runner pipeline runs
// -llvm-request-c-wrappers, which routes `func.call`s through `_mlir_ciface_*`
// symbols. For these scalar-only signatures the C interface is identical to the
// plain entry point, so the forwarders just call through.
VULKAN_WRAPPER_SYMBOL_EXPORT uint64_t _mlir_ciface_mgpuMemAlloc(uint64_t size) {
  return mgpuMemAlloc(size);
}
VULKAN_WRAPPER_SYMBOL_EXPORT void _mlir_ciface_mgpuMemFree(uint64_t address) {
  mgpuMemFree(address);
}
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_mgpuMemcpyHostToDevice(uint64_t dstAddress, uint64_t srcHostPtr,
                                    uint64_t sizeBytes) {
  mgpuMemcpyHostToDevice(dstAddress, srcHostPtr, sizeBytes);
}
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_mgpuMemcpyDeviceToHost(uint64_t dstHostPtr, uint64_t srcAddress,
                                    uint64_t sizeBytes) {
  mgpuMemcpyDeviceToHost(dstHostPtr, srcAddress, sizeBytes);
}

VULKAN_WRAPPER_SYMBOL_EXPORT void *mgpuModuleLoad(const void *data,
                                                  size_t gpuBlobSize) {
  // gpuBlobSize is the size of the data in bytes.
  return new VulkanModule(static_cast<const uint8_t *>(data), gpuBlobSize);
}

VULKAN_WRAPPER_SYMBOL_EXPORT void mgpuModuleUnload(void *vkModule) {
  delete static_cast<VulkanModule *>(vkModule);
}

VULKAN_WRAPPER_SYMBOL_EXPORT void *mgpuModuleGetFunction(void *vkModule,
                                                         const char *name) {
  if (!vkModule)
    abort();
  return static_cast<VulkanModule *>(vkModule)->getFunction(name);
}

// Sentinel placed by the compiler in params[0] to request the explicit
// buffer-device-address (PhysicalStorageBuffer) launch path. The runtime never
// inspects the SPIR-V module to decide this; the calling convention dictates it.
static constexpr uint64_t kVulkanBufferDeviceAddressMagic = 0x4244414d4c524956ULL;

// Sentinel placed by the compiler in params[0] to request the raw-argument
// (CUDA-style) launch path: the kernel's by-value arguments (device addresses
// from mgpuMemAlloc and scalars) are passed directly as push constants, with no
// runtime-managed buffers. params[1] points to the push-constant blob and
// params[2] holds its byte size.
static constexpr uint64_t kVulkanRawArgsMagic = 0x5752414d4c524956ULL;

// MLIR's standard unranked memref descriptor: a rank and a pointer to the
// ranked descriptor { allocatedPtr, alignedPtr, offset, sizes[rank],
// strides[rank] }.
struct UnrankedMemRef {
  int64_t rank;
  void *descriptor;
};

// Handles the buffer-device-address calling convention. The params array is:
//   [0] i64 marker (== kVulkanBufferDeviceAddressMagic)
//   [1 + 2i + 0] ptr  pointer to buffer i's unranked memref descriptor
//   [1 + 2i + 1] i64  buffer i data size in bytes
// The runtime reads each ranked descriptor (MLIR's standard layout) directly:
// it obtains the host data pointer from the descriptor's alignedPtr field and
// the shape/stride/offset metadata for building the device-resident descriptor.
static void launchBufferDeviceAddress(VulkanRuntimeManager *manager,
                                      void **params, size_t paramsCount) {
  const size_t paramsPerBuffer = 2;
  if ((paramsCount - 1) % paramsPerBuffer != 0)
    abort(); // Calling convention mismatch.
  const uint64_t numBuffers = (paramsCount - 1) / paramsPerBuffer;

  const DescriptorSetIndex setIndex = 0;
  for (uint64_t i = 0; i < numBuffers; ++i) {
    const size_t base = 1 + i * paramsPerBuffer;
    const auto *unranked = static_cast<UnrankedMemRef *>(params[base + 0]);
    auto dataByteSize = *static_cast<uint64_t *>(params[base + 1]);

    // The ranked descriptor: { void* allocated; void* aligned; int64_t offset;
    // int64_t sizes[rank]; int64_t strides[rank]; }. The host data pointer is
    // the aligned pointer (second field).
    const auto *rankedBytes = static_cast<const uint8_t *>(unranked->descriptor);
    void *alignedPtr = *reinterpret_cast<void *const *>(rankedBytes +
                                                        sizeof(void *));
    uint32_t descriptorByteSize =
        2 * sizeof(void *) + sizeof(int64_t) +
        2 * static_cast<uint32_t>(unranked->rank) * sizeof(int64_t);

    VulkanHostMemoryBuffer memBuffer{alignedPtr,
                                     static_cast<uint32_t>(dataByteSize)};
    manager->setResourceData(setIndex, static_cast<BindingIndex>(i), memBuffer);
    manager->setMemRefDescriptor(static_cast<BindingIndex>(i),
                                 unranked->descriptor, descriptorByteSize);
  }
}

VULKAN_WRAPPER_SYMBOL_EXPORT void
mgpuLaunchKernel(void *vkKernel, size_t gridX, size_t gridY, size_t gridZ,
                 size_t /*blockX*/, size_t /*blockY*/, size_t /*blockZ*/,
                 size_t /*smem*/, void *vkRuntimeManager, void **params,
                 void ** /*extra*/, size_t paramsCount) {
  auto *manager = static_cast<VulkanRuntimeManager *>(vkRuntimeManager);

  // The raw-argument (CUDA-style) calling convention passes the kernel's
  // by-value arguments as push constants and dispatches on the shared memory
  // context, so the device addresses produced by mgpuMemAlloc are valid. There
  // are no runtime-managed buffers, so this bypasses the VulkanRuntime entirely.
  if (paramsCount >= 1 &&
      *static_cast<uint64_t *>(params[0]) == kVulkanRawArgsMagic) {
    auto *function = static_cast<VulkanFunction *>(vkKernel);
    const auto *pushConstants = *static_cast<uint8_t **>(params[1]);
    auto pushConstantSize = *static_cast<uint64_t *>(params[2]);
    getGlobalVulkanContext().launchCompute(
        function->module->blobData(),
        static_cast<uint32_t>(function->module->blobSizeInBytes()),
        function->name.c_str(), static_cast<uint32_t>(gridX),
        static_cast<uint32_t>(gridY), static_cast<uint32_t>(gridZ),
        pushConstants, static_cast<uint32_t>(pushConstantSize));
    return;
  }

  // The buffer-device-address calling convention is selected by a sentinel in
  // params[0] set by the compiler-emitted launch (an explicit handshake rather
  // than inspecting the shader).
  if (paramsCount >= 1 &&
      *static_cast<uint64_t *>(params[0]) == kVulkanBufferDeviceAddressMagic) {
    launchBufferDeviceAddress(manager, params, paramsCount);
  } else {
    // GpuToLLVMConversionPass with the kernelBarePtrCallConv and
    // kernelIntersperseSizeCallConv options will set up the params array like:
    // { &memref_ptr0, &memref_size0, &memref_ptr1, &memref_size1, ... }
    const size_t paramsPerMemRef = 2;
    if (paramsCount % paramsPerMemRef != 0) {
      abort(); // This would indicate a serious calling convention mismatch.
    }
    const DescriptorSetIndex setIndex = 0;
    BindingIndex bindIndex = 0;
    for (size_t i = 0; i < paramsCount; i += paramsPerMemRef) {
      void *memrefBufferBasePtr = *static_cast<void **>(params[i + 0]);
      size_t memrefBufferSize = *static_cast<size_t *>(params[i + 1]);
      VulkanHostMemoryBuffer memBuffer{memrefBufferBasePtr,
                                       static_cast<uint32_t>(memrefBufferSize)};
      manager->setResourceData(setIndex, bindIndex, memBuffer);
      ++bindIndex;
    }
  }

  manager->setNumWorkGroups(NumWorkGroups{static_cast<uint32_t>(gridX),
                                          static_cast<uint32_t>(gridY),
                                          static_cast<uint32_t>(gridZ)});

  auto *function = static_cast<VulkanFunction *>(vkKernel);
  // Expected size should be in bytes.
  manager->setShaderModule(
      function->module->blobData(),
      static_cast<uint32_t>(function->module->blobSizeInBytes()));
  manager->setEntryPoint(function->name.c_str());

  manager->runOnVulkan();
}

//===----------------------------------------------------------------------===//
//
// Miscellaneous utility functions that can be directly used by tests.
//
//===----------------------------------------------------------------------===//

/// Fills the given 1D float memref with the given float value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource1DFloat(MemRefDescriptor<float, 1> *ptr, // NOLINT
                                 float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0], value);
}

/// Fills the given 2D float memref with the given float value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource2DFloat(MemRefDescriptor<float, 2> *ptr, // NOLINT
                                 float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1], value);
}

/// Fills the given 3D float memref with the given float value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource3DFloat(MemRefDescriptor<float, 3> *ptr, // NOLINT
                                 float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1] * ptr->sizes[2],
              value);
}

/// Fills the given 1D int memref with the given int value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource1DInt(MemRefDescriptor<int32_t, 1> *ptr, // NOLINT
                               int32_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0], value);
}

/// Fills the given 2D int memref with the given int value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource2DInt(MemRefDescriptor<int32_t, 2> *ptr, // NOLINT
                               int32_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1], value);
}

/// Fills the given 3D int memref with the given int value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource3DInt(MemRefDescriptor<int32_t, 3> *ptr, // NOLINT
                               int32_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1] * ptr->sizes[2],
              value);
}

/// Fills the given 1D int memref with the given int8 value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource1DInt8(MemRefDescriptor<int8_t, 1> *ptr, // NOLINT
                                int8_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0], value);
}

/// Fills the given 2D int memref with the given int8 value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource2DInt8(MemRefDescriptor<int8_t, 2> *ptr, // NOLINT
                                int8_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1], value);
}

/// Fills the given 3D int memref with the given int8 value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource3DInt8(MemRefDescriptor<int8_t, 3> *ptr, // NOLINT
                                int8_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1] * ptr->sizes[2],
              value);
}
}

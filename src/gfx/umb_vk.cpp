#include "vulkan/vulkan_core.h"

#include <SDL.h>
#include <chrono>
#include <core/umb_hash_table.h>
#include <functional>
#include <gfx/umb_gfx.h>
#include <utility>

#define VMA_VULKAN_VERSION 1002000
#define VMA_IMPLEMENTATION
#include <gfx/vk_mem_alloc.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <gfx/tiny_obj_loader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <gfx/stb_image.h>

#define VK_CHECK(x, msg)   \
  do {                     \
    if (x != VK_SUCCESS) { \
      printf("%s\n", msg); \
      abort();             \
    }                      \
                           \
  } while (0)

static constexpr u32 MAX_SWAPCHAIN_IMAGES                    = 3;
static constexpr u32 MAX_FRAMES_IN_FLIGHT                    = 2;
static constexpr u32 MAX_DESCRIPTOR_SET_LAYOUTS_PER_PIPELINE = 3;
static constexpr u32 MAX_SHADER_STAGES                       = 3;
static constexpr u32 MAX_GPU_OBJECTS                         = 1000;

static constexpr const char* VALIDATION_LAYERS[] = {
    "VK_LAYER_KHRONOS_validation",
};
static constexpr const char* DEVICE_EXTENSIONS[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME,
#if defined(__APPLE__)
    "VK_KHR_portability_subset",
#endif
};

UMB_CONTAINER_DEF(VkDeviceQueueCreateInfo);
UMB_CONTAINER_DEF(VkSurfaceFormatKHR);
UMB_CONTAINER_DEF(VkPresentModeKHR);
UMB_CONTAINER_DEF(VkVertexInputAttributeDescription);

struct umbvk_queue_family_indices {
  u32 graphics_and_compute_queue_idx = -1;
  u32 present_queue_idx              = -1;
};

struct umbvk_swapchain_support_details {
  VkSurfaceCapabilitiesKHR     capabilities;
  umb_array_VkSurfaceFormatKHR formats;
  umb_array_VkPresentModeKHR   present_modes;
};

struct umb_image_t {
  VkImage       image;
  VmaAllocation allocation;
};

struct umb_texture_t {
  umb_image   image;
  VkImageView image_view;
};

struct umbvk_swapchain {
  b32            initialized;
  u32            current_image;
  u32            n_images;
  VkSwapchainKHR swapchain;
  VkImage        images[MAX_SWAPCHAIN_IMAGES];
  VkImageView    image_views[MAX_SWAPCHAIN_IMAGES];
  VkFramebuffer  framebuffers[MAX_SWAPCHAIN_IMAGES];
  VkSemaphore    image_available_semaphore[MAX_SWAPCHAIN_IMAGES];
  VkExtent2D     extent;
  VkFormat       color_format;

  umb_image_t depth_image;
  VkImageView depth_image_view;
};

typedef u32 umbvk_desc_count[MAX_DESCRIPTOR_SET_LAYOUTS_PER_PIPELINE];

struct umbvk_desc_set_layout {
  VkDescriptorSetLayout layout;
  umbvk_desc_count      counts;
};

struct umbvk_shader_stage {
  VkShaderModule        shader_module;
  VkShaderStageFlagBits stage_bits;
};
UMB_CONTAINER_DEF(umbvk_shader_stage);

struct umbvk_pipeline_builder {
  umb_array_umbvk_shader_stage           shader_stages;
  VkPipelineVertexInputStateCreateInfo   vertex_input_info;
  VkPipelineInputAssemblyStateCreateInfo input_assembly;
  VkPipelineDepthStencilStateCreateInfo  depth_stencil;
  VkViewport                             viewport;
  VkRect2D                               scissor;
  VkPipelineRasterizationStateCreateInfo rasterizer;
  VkPipelineColorBlendAttachmentState    color_blend_attachment;
  VkPipelineMultisampleStateCreateInfo   multisampling;
  VkPipelineDynamicStateCreateInfo       dynamic_state;
  VkPipelineLayout                       pipeline_layout;
};

struct umbvk_buffer {
  VkBuffer      buffer;
  VmaAllocation alloc;
};

struct umb_mesh_t {
  umb_array_umb_mesh_vertex vertices;
  umbvk_buffer              vertex_buffer;
};

struct umb_text_mesh_t {
  umb_array_umb_text_mesh_vertex vertices;
  umbvk_buffer                   vertex_buffer;
};

struct umbvk_cmd_buffer {
  VkCommandBuffer cmd_buff;
  VkCommandPool   cmd_pool;
  umb_pipeline*   active_gfx_pipeline;
  umb_pipeline*   active_cmp_pipeline;
};

struct umbvk_upload_context {
  VkCommandPool   cmd_pool;
  VkCommandBuffer cmd_buff;
  VkFence         upload_fence;
};

struct umb_gpu_camera_data {
  glm::mat4 view;
  glm::mat4 proj;
  glm::mat4 viewproj;
};

struct umb_gpu_scene_data {
  glm::vec4 fog_color;
  glm::vec4 fog_dsitances;
  glm::vec4 ambient_color;
  glm::vec4 sunlight_direction;
  glm::vec4 sunlight_color;
};

struct umb_gpu_object_data {
  glm::mat4 model_matrix;
};

struct umbvk_frame {
  VkSemaphore      image_available_semaphore, render_finished_semaphore;
  VkFence          render_fence;
  umbvk_cmd_buffer cmd;

  umbvk_buffer    camera_buffer;
  VkDescriptorSet global_descriptor;

  umbvk_buffer    object_buffer;
  VkDescriptorSet object_descriptor;
};

class umbvk_deletion_queue {
  public:
  using del_func = std::function<void()>;

  void push(del_func&& function) {
    UMB_ASSERT(n_deletors < MAX_DELETORS);
    deletors[n_deletors++] = function;
  }

  void flush() {
    for (i32 i = n_deletors - 1; i >= 0; --i) { deletors[i](); }
    n_deletors = 0;
  }

  private:
  static constexpr u32 MAX_DELETORS = 1024;
  u32                  n_deletors;
  del_func             deletors[MAX_DELETORS];
};

struct {
  VkInstance               instance;
  VkPhysicalDevice         physical_device;
  VkDevice                 device;
  VkDebugUtilsMessengerEXT debug_messenger;
  VkQueue                  graphics_queue;
  VkQueue                  compute_queue;
  VkQueue                  present_queue;
  b32                      validation_layers_enabled = true;
  b32                      initialized               = false;
  b32                      framebuffer_resized;

  umbvk_queue_family_indices queue_families;

  VkRenderPass    compatible_render_pass;
  VkFormat        depth_format;
  umbvk_swapchain swapchain;
  VkSurfaceKHR    surface;

  u32         frame_id;
  umbvk_frame frames[MAX_FRAMES_IN_FLIGHT];

  VmaAllocator allocator;

  umb_window*          window;
  umb_arena_t          arena;
  umbvk_deletion_queue deletion_queue;

  umb_ptr_array_umb_render_object render_objects;

  VkDescriptorSetLayout global_set_layout;
  VkDescriptorSetLayout object_set_layout;
  VkDescriptorPool      descriptor_pool;

  umb_gpu_scene_data scene_parameters;
  umbvk_buffer       scene_parameters_buffer;

  umbvk_upload_context upload_context;

  umb_hash_table materials;
  umb_hash_table meshes;
  umb_hash_table textures;
} _vk;

struct umb_push_constants {
  glm::vec4 data;
  glm::mat4 render_matrix;
};

u64 umbvk_pad_uniform_buffer_size(u64 original_size) {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(_vk.physical_device, &props);
  u64 min_ubo_alignment = props.limits.minUniformBufferOffsetAlignment;
  u64 aligned_size      = original_size;
  if (min_ubo_alignment > 0) {
    aligned_size = (aligned_size + min_ubo_alignment - 1) & ~(min_ubo_alignment - 1);
  }
  return aligned_size;
}

VkVertexInputBindingDescription umbvk_get_vertex_binding_description() {
  VkVertexInputBindingDescription binding_desc = {
      .binding   = 0,
      .stride    = sizeof(umb_mesh_vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
  };
  return binding_desc;
}

umb_array_VkVertexInputAttributeDescription umbvk_get_vertex_attribute_descriptions() {
  umb_array_VkVertexInputAttributeDescription attribute_descs =
      UMB_ARRAY_CREATE(VkVertexInputAttributeDescription, &_vk.arena, 3);

  VkVertexInputAttributeDescription pos = {
      .binding  = 0,
      .location = 0,
      .format   = VK_FORMAT_R32G32B32_SFLOAT,
      .offset   = offsetof(umb_mesh_vertex, position)};
  UMB_ARRAY_PUSH(attribute_descs, pos);

  VkVertexInputAttributeDescription norm = {
      .binding  = 0,
      .location = 1,
      .format   = VK_FORMAT_R32G32B32_SFLOAT,
      .offset   = offsetof(umb_mesh_vertex, normal)};
  UMB_ARRAY_PUSH(attribute_descs, norm);

  VkVertexInputAttributeDescription col = {
      .binding  = 0,
      .location = 2,
      .format   = VK_FORMAT_R32G32B32_SFLOAT,
      .offset   = offsetof(umb_mesh_vertex, color)};
  UMB_ARRAY_PUSH(attribute_descs, col);

  VkVertexInputAttributeDescription uv = {
      .binding  = 0,
      .location = 3,
      .format   = VK_FORMAT_R32G32_SFLOAT,
      .offset   = offsetof(umb_mesh_vertex, uv)};
  UMB_ARRAY_PUSH(attribute_descs, uv);

  return attribute_descs;
}

umbvk_buffer umbvk_buffer_create_gpu_upload(u64 alloc_size, VkBufferUsageFlags usage) {
  VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size  = alloc_size,
      .usage = usage,
  };

  VmaAllocationCreateInfo alloc_info = {
      .usage         = VMA_MEMORY_USAGE_CPU_TO_GPU,
      .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
  };

  umbvk_buffer buffer;

  VK_CHECK(
      vmaCreateBuffer(
          _vk.allocator,
          &buffer_info,
          &alloc_info,
          &buffer.buffer,
          &buffer.alloc,
          nullptr),
      "Failed to create vertex buffer!");

  _vk.deletion_queue.push([=]() { vmaDestroyBuffer(_vk.allocator, buffer.buffer, buffer.alloc); });

  return buffer;
}

umbvk_buffer umbvk_buffer_create_staging(u64 alloc_size) {
  VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size  = alloc_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
  };

  VmaAllocationCreateInfo alloc_info = {
      .usage         = VMA_MEMORY_USAGE_CPU_ONLY,
      .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
  };

  umbvk_buffer buffer;
  VK_CHECK(
      vmaCreateBuffer(
          _vk.allocator,
          &buffer_info,
          &alloc_info,
          &buffer.buffer,
          &buffer.alloc,
          nullptr),
      "Failed to create vertex buffer!");

  _vk.deletion_queue.push([=]() { vmaDestroyBuffer(_vk.allocator, buffer.buffer, buffer.alloc); });

  return buffer;
}

umbvk_buffer umbvk_buffer_create_transfer(u64 alloc_size, VkBufferUsageFlags usage) {
  VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size  = alloc_size,
      .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  };

  VmaAllocationCreateInfo alloc_info = {.usage = VMA_MEMORY_USAGE_GPU_ONLY};

  umbvk_buffer buffer;
  VK_CHECK(
      vmaCreateBuffer(
          _vk.allocator,
          &buffer_info,
          &alloc_info,
          &buffer.buffer,
          &buffer.alloc,
          nullptr),
      "Failed to create vertex buffer!");

  _vk.deletion_queue.push([=]() { vmaDestroyBuffer(_vk.allocator, buffer.buffer, buffer.alloc); });

  return buffer;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL umbvk_debug_utils_message_fn(
    VkDebugUtilsMessageSeverityFlagBitsEXT      message_severity,
    VkDebugUtilsMessageTypeFlagsEXT             message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void*                                       user_data) {
  printf("validation layer: %s\n\n", callback_data->pMessage);
  return VK_FALSE;
}

void debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& create_info) {
  create_info.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
  create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
  create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
  create_info.pfnUserCallback = umbvk_debug_utils_message_fn, create_info.pUserData = nullptr;
}

VkResult umbvk_create_debug_utils_messenger_ext(
    VkInstance                                instance,
    const VkDebugUtilsMessengerCreateInfoEXT* create_info,
    const VkAllocationCallbacks*              allocator,
    VkDebugUtilsMessengerEXT*                 debug_messenger) {
  auto create_func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance,
      "vkCreateDebugUtilsMessengerEXT");
  if (create_func != nullptr)
    return create_func(instance, create_info, allocator, debug_messenger);
  else
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void umbvk_destroy_debug_utils_messenger_ext(
    VkInstance                   instance,
    VkDebugUtilsMessengerEXT     debug_messenger,
    const VkAllocationCallbacks* allocator) {
  auto destroy_func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance,
      "vkDestroyDebugUtilsMessengerEXT");
  if (destroy_func != nullptr) destroy_func(instance, debug_messenger, allocator);
}

umbvk_swapchain_support_details
umbvk_query_swapchain_support(VkSurfaceKHR surface, VkPhysicalDevice physical_device) {
  umbvk_swapchain_support_details details {};

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &details.capabilities);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &details.formats.len, nullptr);
  details.formats.data = umb_arena_push_array(&_vk.arena, VkSurfaceFormatKHR, details.formats.len);
  if (details.formats.len != 0)
    vkGetPhysicalDeviceSurfaceFormatsKHR(
        physical_device,
        surface,
        &details.formats.len,
        details.formats.data);

  vkGetPhysicalDeviceSurfacePresentModesKHR(
      physical_device,
      surface,
      &details.present_modes.len,
      nullptr);
  details.present_modes.data =
      umb_arena_push_array(&_vk.arena, VkPresentModeKHR, details.present_modes.len);
  if (details.present_modes.len != 0)
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device,
        surface,
        &details.present_modes.len,
        details.present_modes.data);

  return details;
}

VkSurfaceFormatKHR
umbvk_choose_swap_surface_format(umb_array_VkSurfaceFormatKHR available_formats) {
  for (i32 i = 0; i < available_formats.len; i++) {
    VkSurfaceFormatKHR available_format = available_formats.data[i];
    if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return available_format;
    }
  }
  return available_formats.data[0];
}

VkPresentModeKHR umbvk_choose_swap_present_mode(umb_array_VkPresentModeKHR present_modes) {
  for (i32 i = 0; i < present_modes.len; i++) {
    VkPresentModeKHR available_present_mode = present_modes.data[i];
    if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) { return available_present_mode; }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D
umbvk_choose_swap_extent(umb_window* window, const VkSurfaceCapabilitiesKHR capabilities) {
  if (capabilities.currentExtent.width != UINT32_MAX) return capabilities.currentExtent;

  i32 width, height;
  SDL_Vulkan_GetDrawableSize((SDL_Window*)window->raw_handle, &width, &height);

  VkExtent2D actual_extent = {static_cast<u32>(width), static_cast<u32>(height)};

  actual_extent.width = UMB_CLAMP(
      actual_extent.width,
      capabilities.minImageExtent.width,
      capabilities.maxImageExtent.width);
  actual_extent.height = UMB_CLAMP(
      actual_extent.height,
      capabilities.minImageExtent.height,
      capabilities.maxImageExtent.height);

  return actual_extent;
}

umbvk_queue_family_indices
umbvk_find_queue_families(VkSurfaceKHR surface, VkPhysicalDevice phys_device) {
  umb_scope_arena scope(&_vk.arena);

  umbvk_queue_family_indices indices;

  u32 queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_family_count, nullptr);
  VkQueueFamilyProperties* queue_families =
      umb_arena_push_array(&_vk.arena, VkQueueFamilyProperties, queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_family_count, queue_families);

  for (i32 i = 0; i < queue_family_count; i++) {
    if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT &&
        queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
      indices.graphics_and_compute_queue_idx = i;

    VkBool32 present_support = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(phys_device, i, surface, &present_support);
    if (present_support) indices.present_queue_idx = i;

    if (indices.graphics_and_compute_queue_idx != -1 && indices.present_queue_idx != -1) break;
  }

  return indices;
}

VkRenderPass umbvk_create_render_pass(VkFormat image_format) {
  VkAttachmentDescription color_attachment {
      .format         = image_format,
      .samples        = VK_SAMPLE_COUNT_1_BIT,
      .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
  };

  VkAttachmentReference color_attachment_ref {
      .attachment = 0,
      .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
  };

  VkAttachmentDescription depth_attachment = {
      .format         = _vk.depth_format,
      .samples        = VK_SAMPLE_COUNT_1_BIT,
      .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
  };

  VkAttachmentReference depth_attachment_ref = {
      .attachment = 1,
      .layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
  };

  VkAttachmentDescription attachments[2] {color_attachment, depth_attachment};

  VkSubpassDescription subpass {
      .pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount    = 1,
      .pColorAttachments       = &color_attachment_ref,
      .pDepthStencilAttachment = &depth_attachment_ref,
  };

  VkSubpassDependency dependency {
      .srcSubpass    = VK_SUBPASS_EXTERNAL,
      .dstSubpass    = 0,
      .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
  };

  VkSubpassDependency depth_dependency {
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask =
          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      .dstStageMask =
          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
  };

  VkSubpassDependency dependencies[2] = {dependency, depth_dependency};

  VkRenderPass           render_pass;
  VkRenderPassCreateInfo render_pass_info {
      .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2,
      .pAttachments    = attachments,
      .subpassCount    = 1,
      .pSubpasses      = &subpass,
      .dependencyCount = 2,
      .pDependencies   = dependencies,
  };

  VK_CHECK(
      vkCreateRenderPass(_vk.device, &render_pass_info, nullptr, &render_pass),
      "failed to create render pass!");

  _vk.deletion_queue.push([=]() { vkDestroyRenderPass(_vk.device, render_pass, nullptr); });

  return render_pass;
}

umb_array_str umbvk_get_required_extensions(umb_window* window) {
  u32 sdl_extension_count = 0;
  SDL_Vulkan_GetInstanceExtensions((SDL_Window*)window->raw_handle, &sdl_extension_count, nullptr);

  umb_array_str required_extensions =
      UMB_ARRAY_CREATE(str, &_vk.arena, (sdl_extension_count + 5) * 128);

  const char** sdl_extensions = umb_arena_push_array(&_vk.arena, const char*, sdl_extension_count);
  SDL_Vulkan_GetInstanceExtensions(
      (SDL_Window*)window->raw_handle,
      &sdl_extension_count,
      sdl_extensions);
  for (i32 i = 0; i < sdl_extension_count; i++) {
    UMB_ARRAY_PUSH(required_extensions, sdl_extensions[i]);
  }

  if (_vk.validation_layers_enabled)
    UMB_ARRAY_PUSH(required_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  UMB_ARRAY_PUSH(required_extensions, "VK_KHR_get_physical_device_properties2");

#if defined(__APPLE__)
  UMB_ARRAY_PUSH(required_extensions, "VK_KHR_portability_enumeration");
#endif

  return required_extensions;
}

b32 umbvk_check_validation_layer_support(const char* const* validation_layers) {
  umb_scope_arena scope(&_vk.arena);

  u32 available_layer_count;
  vkEnumerateInstanceLayerProperties(&available_layer_count, nullptr);
  VkLayerProperties* available_layers =
      umb_arena_push_array(&_vk.arena, VkLayerProperties, available_layer_count);
  vkEnumerateInstanceLayerProperties(&available_layer_count, available_layers);

  i32 validation_layer_count = UMB_ARRAY_COUNT(validation_layers, str);
  for (i32 i = 0; i < validation_layer_count; i++) {
    b32 layer_found = false;
    for (i32 j = 0; j < available_layer_count; j++) {
      if (strcmp(validation_layers[i], available_layers[j].layerName) == 0) {
        layer_found = true;
        break;
      }
    }
    if (!layer_found) return false;
  }

  return true;
}

b32 umbvk_check_device_extension_support(
    VkPhysicalDevice   phys_device,
    const char* const* device_extensions) {
  umb_scope_arena scope(&_vk.arena);

  u32 extension_count;
  vkEnumerateDeviceExtensionProperties(phys_device, nullptr, &extension_count, nullptr);
  VkExtensionProperties* available_extensions =
      umb_arena_push_array(&_vk.arena, VkExtensionProperties, extension_count);
  vkEnumerateDeviceExtensionProperties(
      phys_device,
      nullptr,
      &extension_count,
      available_extensions);

  u32 n_requested_exts = UMB_ARRAY_COUNT(device_extensions, str);

  u32 n_found_exts = 0;
  for (i32 i = 0; i < n_requested_exts; ++i) {
    for (i32 j = 0; j < extension_count; ++j) {
      const VkExtensionProperties* ext = &available_extensions[j];
      if (strcmp(device_extensions[i], ext->extensionName) == 0) {
        n_found_exts++;
        continue;
      }
    }
  }

  return n_found_exts == n_requested_exts;
}

b32 umbvk_is_device_suitable(
    VkSurfaceKHR       surface,
    VkPhysicalDevice   phys_device,
    const char* const* device_extensions) {
  umbvk_queue_family_indices indices = umbvk_find_queue_families(surface, phys_device);
  b32 extensions_supported = umbvk_check_device_extension_support(phys_device, device_extensions);

  bool swap_chain_adequate = false;
  if (extensions_supported) {
    umbvk_swapchain_support_details swap_chain_support =
        umbvk_query_swapchain_support(surface, phys_device);
    swap_chain_adequate =
        swap_chain_support.formats.len >= 0 && swap_chain_support.present_modes.len >= 0;
  }

  return indices.present_queue_idx != -1 && indices.graphics_and_compute_queue_idx != -1 &&
         extensions_supported && swap_chain_adequate;
}

void umbvk_set_physical_device() {
  umb_scope_arena scope(&_vk.arena);

  u32 device_count = 0;
  vkEnumeratePhysicalDevices(_vk.instance, &device_count, nullptr);
  UMB_ASSERT(device_count > 0);

  VkPhysicalDevice* devices = umb_arena_push_array(&_vk.arena, VkPhysicalDevice, device_count);
  vkEnumeratePhysicalDevices(_vk.instance, &device_count, devices);

  for (i32 i = 0; i < device_count; i++) {
    VkPhysicalDevice prospective_device = devices[i];
    if (umbvk_is_device_suitable(_vk.surface, prospective_device, DEVICE_EXTENSIONS)) {
      _vk.physical_device = prospective_device;
      break;
    }
  }
  UMB_ASSERT(_vk.physical_device != VK_NULL_HANDLE);
  _vk.queue_families = umbvk_find_queue_families(_vk.surface, _vk.physical_device);
}

void umbvk_set_logical_device() {
  umb_scope_arena scope(&_vk.arena);

  umb_array_VkDeviceQueueCreateInfo queue_create_infos =
      UMB_ARRAY_CREATE(VkDeviceQueueCreateInfo, &_vk.arena, 8);

  u32 unique_qfam[3];
  u32 unique_qfam_count            = 0;
  unique_qfam[unique_qfam_count++] = _vk.queue_families.graphics_and_compute_queue_idx;
  if (_vk.queue_families.graphics_and_compute_queue_idx != _vk.queue_families.present_queue_idx) {
    unique_qfam[unique_qfam_count++] = _vk.queue_families.present_queue_idx;
  }

  f32 queue_priority = 1.0f;
  for (i32 i = 0; i < unique_qfam_count; ++i) {
    VkDeviceQueueCreateInfo queue_create_info {
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = unique_qfam[i],
        .queueCount       = 1,
        .pQueuePriorities = &queue_priority,
    };
    UMB_ARRAY_PUSH(queue_create_infos, queue_create_info);
  }

  VkPhysicalDeviceFeatures device_features {};
  VkDeviceCreateInfo       create_info {
            .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount    = static_cast<u32>(queue_create_infos.len),
            .pQueueCreateInfos       = queue_create_infos.data,
            .enabledExtensionCount   = UMB_ARRAY_COUNT(DEVICE_EXTENSIONS, str),
            .ppEnabledExtensionNames = DEVICE_EXTENSIONS,
            .pEnabledFeatures        = &device_features,
  };

  if (_vk.validation_layers_enabled) {
    create_info.enabledLayerCount   = UMB_ARRAY_COUNT(VALIDATION_LAYERS, str);
    create_info.ppEnabledLayerNames = VALIDATION_LAYERS;
  } else {
    create_info.enabledLayerCount = 0;
  }

  VK_CHECK(
      vkCreateDevice(_vk.physical_device, &create_info, nullptr, &_vk.device),
      "failed to create logical device!");

  vkGetDeviceQueue(
      _vk.device,
      _vk.queue_families.graphics_and_compute_queue_idx,
      0,
      &_vk.graphics_queue);
  vkGetDeviceQueue(
      _vk.device,
      _vk.queue_families.graphics_and_compute_queue_idx,
      0,
      &_vk.compute_queue);
  vkGetDeviceQueue(_vk.device, _vk.queue_families.present_queue_idx, 0, &_vk.present_queue);
}

void umbvk_destroy_swapchain(umbvk_swapchain* swapchain) {
  if (!swapchain || !swapchain->initialized) return;

  for (i32 i = 0; i < swapchain->n_images; i++) {
    if (swapchain->image_views[i])
      vkDestroyImageView(_vk.device, swapchain->image_views[i], nullptr);
    if (swapchain->framebuffers[i])
      vkDestroyFramebuffer(_vk.device, swapchain->framebuffers[i], nullptr);
  }

  // TODO(brysonm): destroy depth image

  vkDestroySwapchainKHR(_vk.device, swapchain->swapchain, nullptr);

  swapchain->initialized = false;
}

umbvk_swapchain umbvk_create_swapchain(
    umbvk_swapchain_support_details swapchain_support,
    VkSurfaceFormatKHR              surface_format,
    VkPresentModeKHR                present_mode,
    VkExtent2D                      extent) {
  // Swapchain
  u32 image_count = swapchain_support.capabilities.minImageCount + 1;
  if (swapchain_support.capabilities.maxImageCount > 0 &&
      image_count > swapchain_support.capabilities.maxImageCount)
    image_count = swapchain_support.capabilities.maxImageCount;

  VkSwapchainCreateInfoKHR create_info {
      .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface          = _vk.surface,
      .minImageCount    = image_count,
      .imageFormat      = surface_format.format,
      .imageColorSpace  = surface_format.colorSpace,
      .imageExtent      = extent,
      .imageArrayLayers = 1,
      .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .preTransform     = swapchain_support.capabilities.currentTransform,
      .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode      = present_mode,
      .clipped          = VK_TRUE,
      .oldSwapchain     = VK_NULL_HANDLE,
  };

  u32 queue_family_indices[] = {
      _vk.queue_families.graphics_and_compute_queue_idx,
      _vk.queue_families.present_queue_idx};

  if (_vk.queue_families.graphics_and_compute_queue_idx != _vk.queue_families.present_queue_idx) {
    create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices   = queue_family_indices;
  } else {
    create_info.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices   = nullptr;
  }

  VkSwapchainKHR swapchain;
  VK_CHECK(
      vkCreateSwapchainKHR(_vk.device, &create_info, nullptr, &swapchain),
      "failed to create swap chain");

  vkGetSwapchainImagesKHR(_vk.device, swapchain, &image_count, nullptr);
  umbvk_swapchain new_swapchain {
      .n_images     = image_count,
      .swapchain    = swapchain,
      .extent       = extent,
      .color_format = surface_format.format,

  };
  vkGetSwapchainImagesKHR(_vk.device, swapchain, &image_count, new_swapchain.images);

  // Image Views
  for (i32 i = 0; i < image_count; i++) {
    VkImageViewCreateInfo view_create_info {
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = new_swapchain.images[i],
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = new_swapchain.color_format,
        .components =
            {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
        .subresourceRange =
            {
                .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
    };
    VK_CHECK(
        vkCreateImageView(_vk.device, &view_create_info, nullptr, &new_swapchain.image_views[i]),
        "failed to create image views!");
  }

  // Depth Image
  VkExtent3D depth_image_extent = {
      .width  = extent.width,
      .height = extent.height,
      .depth  = 1,
  };

  VkImageCreateInfo dimg_info = {
      .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType   = VK_IMAGE_TYPE_2D,
      .format      = _vk.depth_format,
      .extent      = depth_image_extent,
      .mipLevels   = 1,
      .arrayLayers = 1,
      .samples     = VK_SAMPLE_COUNT_1_BIT,
      .tiling      = VK_IMAGE_TILING_OPTIMAL,
      .usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
  };

  VmaAllocationCreateInfo dimg_allocinfo = {
      .usage         = VMA_MEMORY_USAGE_GPU_ONLY,
      .requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
  };
  vmaCreateImage(
      _vk.allocator,
      &dimg_info,
      &dimg_allocinfo,
      &new_swapchain.depth_image.image,
      &new_swapchain.depth_image.allocation,
      nullptr);

  VkImageViewCreateInfo dimg_view_info = {
      .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .image    = new_swapchain.depth_image.image,
      .format   = _vk.depth_format,
      .subresourceRange =
          {
              .aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT,
              .baseMipLevel   = 0,
              .levelCount     = 1,
              .baseArrayLayer = 0,
              .layerCount     = 1,
          },
  };
  VK_CHECK(
      vkCreateImageView(_vk.device, &dimg_view_info, nullptr, &new_swapchain.depth_image_view),
      "failed to create depth image view!");

  _vk.deletion_queue.push([=]() {
    vkDestroyImageView(_vk.device, new_swapchain.depth_image_view, nullptr);
    vmaDestroyImage(
        _vk.allocator,
        new_swapchain.depth_image.image,
        new_swapchain.depth_image.allocation);
  });

  // Framebuffers
  for (i32 i = 0; i < image_count; i++) {
    VkImageView attachments[] = {
        new_swapchain.image_views[i],
        new_swapchain.depth_image_view,
    };

    VkFramebufferCreateInfo framebuffer_info {
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass      = _vk.compatible_render_pass,
        .attachmentCount = 2,
        .pAttachments    = attachments,
        .width           = new_swapchain.extent.width,
        .height          = new_swapchain.extent.height,
        .layers          = 1,
    };

    VK_CHECK(
        vkCreateFramebuffer(_vk.device, &framebuffer_info, nullptr, &new_swapchain.framebuffers[i]),
        "failed to create framebuffer!");
  }

  new_swapchain.initialized = true;
  return new_swapchain;
}

void umbvk_recreate_swapchain() {
  i32 width = 0, height = 0;

  SDL_Vulkan_GetDrawableSize((SDL_Window*)_vk.window->raw_handle, &width, &height);
  while (width == 0 || height == 0) {
    SDL_Vulkan_GetDrawableSize((SDL_Window*)_vk.window->raw_handle, &width, &height);
    SDL_WaitEvent(nullptr);
  }

  vkDeviceWaitIdle(_vk.device);

  umbvk_destroy_swapchain(&_vk.swapchain);

  umbvk_swapchain_support_details swapchain_support =
      umbvk_query_swapchain_support(_vk.surface, _vk.physical_device);
  VkSurfaceFormatKHR surface_format = umbvk_choose_swap_surface_format(swapchain_support.formats);
  VkPresentModeKHR   present_mode = umbvk_choose_swap_present_mode(swapchain_support.present_modes);
  VkExtent2D         extent = umbvk_choose_swap_extent(_vk.window, swapchain_support.capabilities);
  _vk.swapchain = umbvk_create_swapchain(swapchain_support, surface_format, present_mode, extent);
}

umbvk_shader_stage
umbvk_shader_stage_create(const umb_array_byte code, VkShaderStageFlagBits stage_bits) {
  VkShaderModuleCreateInfo create_info {
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code.len,
      .pCode    = reinterpret_cast<const u32*>(code.data),
  };

  VkShaderModule shader_module;
  VK_CHECK(
      vkCreateShaderModule(_vk.device, &create_info, nullptr, &shader_module),
      "failed to create shader module!");

  return umbvk_shader_stage {
      .shader_module = shader_module,
      .stage_bits    = stage_bits,
  };
}

void umbvk_shader_stage_destroy(umbvk_shader_stage* shader) {
  vkDestroyShaderModule(_vk.device, shader->shader_module, nullptr);
}

umb_pipeline
umbvk_pipeline_builder_build(umbvk_pipeline_builder* builder, VkDevice device, VkRenderPass pass) {
  VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = nullptr,

      .viewportCount = 1,
      .pViewports    = &builder->viewport,
      .scissorCount  = 1,
      .pScissors     = &builder->scissor,
  };

  VkPipelineColorBlendStateCreateInfo color_blending = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .pNext = nullptr,

      .logicOpEnable   = VK_FALSE,
      .logicOp         = VK_LOGIC_OP_COPY,
      .attachmentCount = 1,
      .pAttachments    = &builder->color_blend_attachment,
  };

  VkPipelineShaderStageCreateInfo* shader_stage_create_infos =
      umb_arena_push_array(&_vk.arena, VkPipelineShaderStageCreateInfo, builder->shader_stages.len);

  for (i32 i = 0; i < builder->shader_stages.len; ++i) {
    shader_stage_create_infos[i] = {
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage  = builder->shader_stages.data[i].stage_bits,
        .module = builder->shader_stages.data[i].shader_module,
        .pName  = "main",
    };
  }

  VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext               = nullptr,
      .stageCount          = builder->shader_stages.len,
      .pStages             = shader_stage_create_infos,
      .pVertexInputState   = &builder->vertex_input_info,
      .pInputAssemblyState = &builder->input_assembly,
      .pViewportState      = &viewport_state,
      .pRasterizationState = &builder->rasterizer,
      .pMultisampleState   = &builder->multisampling,
      .pColorBlendState    = &color_blending,
      .pDepthStencilState  = &builder->depth_stencil,
      .layout              = builder->pipeline_layout,
      .renderPass          = pass,
      .subpass             = 0,
      .basePipelineHandle  = VK_NULL_HANDLE,
      .pDynamicState       = &builder->dynamic_state,
  };

  umb_pipeline new_pipeline = {};
  VK_CHECK(
      vkCreateGraphicsPipelines(
          device,
          VK_NULL_HANDLE,
          1,
          &pipeline_info,
          nullptr,
          &new_pipeline.pipeline),
      "Failed to create pipeline!");
  new_pipeline.pipeline_layout = builder->pipeline_layout;

  _vk.deletion_queue.push([=]() {
    vkDestroyPipeline(_vk.device, new_pipeline.pipeline, nullptr);
    vkDestroyPipelineLayout(_vk.device, new_pipeline.pipeline_layout, nullptr);
  });

  return new_pipeline;
}

umbvk_pipeline_builder umbvk_pipeline_builder_create() {
  umbvk_pipeline_builder builder = {};
  builder.shader_stages = UMB_ARRAY_CREATE(umbvk_shader_stage, &_vk.arena, MAX_SHADER_STAGES);
  return builder;
}

umb_pipeline umbvk_default_graphics_pipeline_create() {
  umbvk_pipeline_builder builder = umbvk_pipeline_builder_create();

  umb_array_byte vert_shader_code =
      umb_read_file_binary(&_vk.arena, "res/shaders/basic_shader.vert.spv");
  umbvk_shader_stage vert_stage =
      umbvk_shader_stage_create(vert_shader_code, VK_SHADER_STAGE_VERTEX_BIT);

  umb_array_byte frag_shader_code =
      umb_read_file_binary(&_vk.arena, "res/shaders/basic_shader.frag.spv");
  umbvk_shader_stage frag_stage =
      umbvk_shader_stage_create(frag_shader_code, VK_SHADER_STAGE_FRAGMENT_BIT);

  UMB_ARRAY_PUSH(builder.shader_stages, vert_stage);
  UMB_ARRAY_PUSH(builder.shader_stages, frag_stage);

  VkVertexInputBindingDescription             binding_desc = umbvk_get_vertex_binding_description();
  umb_array_VkVertexInputAttributeDescription attribute_descs =
      umbvk_get_vertex_attribute_descriptions();
  builder.vertex_input_info = {
      .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount   = 1,
      .pVertexBindingDescriptions      = &binding_desc,
      .vertexAttributeDescriptionCount = attribute_descs.len,
      .pVertexAttributeDescriptions    = attribute_descs.data,
  };

  builder.input_assembly = {
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE,
  };

  builder.viewport = {
      .x        = 0.0f,
      .y        = 0.0f,
      .width    = (f32)_vk.swapchain.extent.width,
      .height   = (f32)_vk.swapchain.extent.height,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };

  builder.scissor = {
      .offset = {0, 0},
      .extent = _vk.swapchain.extent,
  };

  builder.depth_stencil = {
      .sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .pNext                 = NULL,
      .depthTestEnable       = VK_TRUE,
      .depthWriteEnable      = VK_TRUE,
      .depthCompareOp        = VK_COMPARE_OP_LESS_OR_EQUAL,
      .depthBoundsTestEnable = VK_FALSE,
      .minDepthBounds        = 0.0f,
      .maxDepthBounds        = 1.0f,
      .stencilTestEnable     = VK_FALSE,
  };

  VkDynamicState dynamic_states[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamic_state = {
      .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = UMB_ARRAY_COUNT(dynamic_states, VkDynamicState),
      .pDynamicStates    = dynamic_states};
  builder.dynamic_state = dynamic_state;

  builder.rasterizer = {
      .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable        = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode             = VK_POLYGON_MODE_FILL,
      .cullMode                = VK_CULL_MODE_BACK_BIT,
      .frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .depthBiasEnable         = VK_FALSE,
      .depthBiasConstantFactor = 0.0f,
      .depthBiasClamp          = 0.0f,
      .depthBiasSlopeFactor    = 0.0f,
      .lineWidth               = 1.0f,
  };

  builder.multisampling = {
      .sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable   = VK_FALSE,
      .minSampleShading      = 1.0f,
      .pSampleMask           = nullptr,
      .alphaToCoverageEnable = VK_FALSE,
      .alphaToOneEnable      = VK_FALSE,
  };

  builder.color_blend_attachment = {
      .blendEnable    = VK_FALSE,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };

  VkPushConstantRange push_constant = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      .offset     = 0,
      .size       = sizeof(umb_push_constants),
  };

  VkDescriptorSetLayout      set_layouts[] = {_vk.global_set_layout, _vk.object_set_layout};
  VkPipelineLayoutCreateInfo pipeline_layout_create_info = {
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &push_constant,
      .setLayoutCount         = UMB_ARRAY_COUNT(set_layouts, VkDescriptorSetLayout),
      .pSetLayouts            = set_layouts,
  };

  VK_CHECK(
      vkCreatePipelineLayout(
          _vk.device,
          &pipeline_layout_create_info,
          nullptr,
          &builder.pipeline_layout),
      "Failed to create pipeline layout.");

  umb_pipeline gfx_pipeline =
      umbvk_pipeline_builder_build(&builder, _vk.device, _vk.compatible_render_pass);

  umbvk_shader_stage_destroy(&vert_stage);
  umbvk_shader_stage_destroy(&frag_stage);

  return gfx_pipeline;
}

void umb_pipeline_destroy(umb_pipeline* pipeline) {
  vkDestroyPipeline(_vk.device, pipeline->pipeline, nullptr);
}

VkDescriptorSetLayoutBinding umbvk_descriptor_set_layout_binding_create(
    VkDescriptorType   type,
    VkShaderStageFlags stages,
    u32                binding) {
  VkDescriptorSetLayoutBinding bind = {
      .binding         = binding,
      .descriptorCount = 1,
      .descriptorType  = type,
      .stageFlags      = stages,
  };
  return bind;
}

VkWriteDescriptorSet umbvk_descriptor_buffer_write(
    VkDescriptorType        type,
    VkDescriptorSet         dst_set,
    VkDescriptorBufferInfo* buffer_info,
    u32                     binding) {
  VkWriteDescriptorSet set_write = {
      .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstBinding      = binding,
      .dstSet          = dst_set,
      .descriptorCount = 1,
      .descriptorType  = type,
      .pBufferInfo     = buffer_info,
  };
  return set_write;
}

void umbvk_set_descriptors() {
  const u64 scene_param_buffer_size =
      MAX_FRAMES_IN_FLIGHT * umbvk_pad_uniform_buffer_size(sizeof(umb_gpu_scene_data));
  _vk.scene_parameters_buffer =
      umbvk_buffer_create_gpu_upload(scene_param_buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

  VkDescriptorPoolSize sizes[] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10},
  };

  VkDescriptorPoolCreateInfo pool_info = {
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets       = 10,
      .poolSizeCount = (u32)UMB_ARRAY_COUNT(sizes, VkDescriptorPoolSize),
      .pPoolSizes    = sizes,
  };
  vkCreateDescriptorPool(_vk.device, &pool_info, nullptr, &_vk.descriptor_pool);

  VkDescriptorSetLayoutBinding cam_bind = umbvk_descriptor_set_layout_binding_create(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      VK_SHADER_STAGE_VERTEX_BIT,
      0);
  VkDescriptorSetLayoutBinding scene_bind = umbvk_descriptor_set_layout_binding_create(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      1);

  VkDescriptorSetLayoutBinding    bindings[] = {cam_bind, scene_bind};
  VkDescriptorSetLayoutCreateInfo set_info   = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = UMB_ARRAY_COUNT(bindings, VkDescriptorSetLayoutBinding),
        .pBindings    = bindings,
  };
  vkCreateDescriptorSetLayout(_vk.device, &set_info, nullptr, &_vk.global_set_layout);

  VkDescriptorSetLayoutBinding obj_bind = umbvk_descriptor_set_layout_binding_create(
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_SHADER_STAGE_VERTEX_BIT,
      0);
  VkDescriptorSetLayoutCreateInfo obj_info = {
      .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings    = &obj_bind,
  };
  vkCreateDescriptorSetLayout(_vk.device, &obj_info, nullptr, &_vk.object_set_layout);

  for (i32 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    _vk.frames[i].object_buffer = umbvk_buffer_create_gpu_upload(
        (sizeof(umb_gpu_object_data) * MAX_GPU_OBJECTS),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    _vk.frames[i].camera_buffer = umbvk_buffer_create_gpu_upload(
        sizeof(umb_gpu_camera_data),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = _vk.descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &_vk.global_set_layout,
    };
    vkAllocateDescriptorSets(_vk.device, &alloc_info, &_vk.frames[i].global_descriptor);

    VkDescriptorSetAllocateInfo obj_alloc_info = {
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = _vk.descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &_vk.object_set_layout,
    };
    vkAllocateDescriptorSets(_vk.device, &obj_alloc_info, &_vk.frames[i].object_descriptor);

    VkDescriptorBufferInfo cam_binfo = {
        .buffer = _vk.frames[i].camera_buffer.buffer,
        .range  = sizeof(umb_gpu_camera_data),
    };
    VkDescriptorBufferInfo scene_binfo = {
        .buffer = _vk.scene_parameters_buffer.buffer,
        .range  = sizeof(umb_gpu_scene_data),
    };
    VkDescriptorBufferInfo obj_binfo = {
        .buffer = _vk.frames[i].object_buffer.buffer,
        .range  = sizeof(umb_gpu_object_data) * MAX_GPU_OBJECTS,
    };

    VkWriteDescriptorSet cam_write = umbvk_descriptor_buffer_write(
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        _vk.frames[i].global_descriptor,
        &cam_binfo,
        0);
    VkWriteDescriptorSet scene_write = umbvk_descriptor_buffer_write(
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        _vk.frames[i].global_descriptor,
        &scene_binfo,
        1);

    VkWriteDescriptorSet obj_write = umbvk_descriptor_buffer_write(
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        _vk.frames[i].object_descriptor,
        &obj_binfo,
        0);

    VkWriteDescriptorSet set_writes[] = {cam_write, scene_write, obj_write};
    vkUpdateDescriptorSets(
        _vk.device,
        UMB_ARRAY_COUNT(set_writes, VkWriteDescriptorSet),
        set_writes,
        0,
        nullptr);
  }

  _vk.deletion_queue.push([&]() {
    vkDestroyDescriptorSetLayout(_vk.device, _vk.global_set_layout, nullptr);
    vkDestroyDescriptorSetLayout(_vk.device, _vk.object_set_layout, nullptr);
    vkDestroyDescriptorPool(_vk.device, _vk.descriptor_pool, nullptr);
  });
}

umbvk_upload_context umbvk_upload_context_create() {
  umbvk_upload_context ctx = {};

  umbvk_queue_family_indices indices = _vk.queue_families;

  VkCommandPoolCreateInfo cmd_pool_info = {
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = indices.graphics_and_compute_queue_idx,
  };

  VK_CHECK(
      vkCreateCommandPool(_vk.device, &cmd_pool_info, nullptr, &ctx.cmd_pool),
      "Failed to create command pool!");

  _vk.deletion_queue.push([=]() { vkDestroyCommandPool(_vk.device, ctx.cmd_pool, nullptr); });

  VkCommandBufferAllocateInfo alloc_info = {
      .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool        = ctx.cmd_pool,
      .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };

  VK_CHECK(
      vkAllocateCommandBuffers(_vk.device, &alloc_info, &ctx.cmd_buff),
      "Failed to allocate command buffer!");

  return ctx;
}

umbvk_cmd_buffer umbvk_cmd_buffer_create() {
  umbvk_cmd_buffer cmd = {};

  umbvk_queue_family_indices indices = _vk.queue_families;

  VkCommandPoolCreateInfo cmd_pool_info = {
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = indices.graphics_and_compute_queue_idx,
  };

  VK_CHECK(
      vkCreateCommandPool(_vk.device, &cmd_pool_info, nullptr, &cmd.cmd_pool),
      "Failed to create command pool!");

  _vk.deletion_queue.push([=]() { vkDestroyCommandPool(_vk.device, cmd.cmd_pool, nullptr); });

  VkCommandBufferAllocateInfo alloc_info = {
      .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool        = cmd.cmd_pool,
      .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };

  VK_CHECK(
      vkAllocateCommandBuffers(_vk.device, &alloc_info, &cmd.cmd_buff),
      "Failed to allocate command buffer!");

  return cmd;
}

void umbvk_cmd_buffer_destroy(umbvk_cmd_buffer* cmd) {
  vkDestroyCommandPool(_vk.device, cmd->cmd_pool, nullptr);
}

void umbvk_cmd_begin(umbvk_cmd_buffer* cmd) {
  VkCommandBufferBeginInfo begin_info = {
      .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags            = 0,
      .pInheritanceInfo = nullptr,
  };

  VK_CHECK(
      vkBeginCommandBuffer(cmd->cmd_buff, &begin_info),
      "Failed to beging recording command buffer!");
}

void umbvk_cmd_bind_graphics_pipeline(umbvk_cmd_buffer* cmd, umb_pipeline* pipeline) {
  // TODO(brysonm): handle previously bound resources
  cmd->active_gfx_pipeline = pipeline;
  vkCmdBindPipeline(cmd->cmd_buff, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);
}

void umbvk_cmd_bind_compute_pipeline(umbvk_cmd_buffer* cmd, umb_pipeline* pipeline) {
  // TODO(brysonm): handle previously bound resources
  cmd->active_cmp_pipeline = pipeline;
  vkCmdBindPipeline(cmd->cmd_buff, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
}

void umbvk_cmd_bind_vertex_buffer(
    umbvk_cmd_buffer* cmd,
    u32               first_binding,
    u32               n_bindings,
    const VkBuffer*   p_buffers,
    VkDeviceSize*     offset) {
  vkCmdBindVertexBuffers(cmd->cmd_buff, first_binding, n_bindings, p_buffers, offset);
}

void umbvk_cmd_push_constants(
    umbvk_cmd_buffer*   cmd,
    umb_push_constants* constants,
    VkShaderStageFlags  shader_stages) {
  vkCmdPushConstants(
      cmd->cmd_buff,
      cmd->active_gfx_pipeline->pipeline_layout,
      shader_stages,
      0,
      sizeof(umb_push_constants),
      constants);
}

void umbvk_cmd_render_pass_begin(umbvk_cmd_buffer* cmd, u32 image_idx) {
  VkClearValue clear_color    = {{0.0f, 0.0f, 0.0f, 1.0f}};
  VkClearValue depth_clear    = {.depthStencil = 1.f};
  VkClearValue clear_values[] = {clear_color, depth_clear};

  VkRenderPassBeginInfo render_pass_info = {
      .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass  = _vk.compatible_render_pass,
      .framebuffer = _vk.swapchain.framebuffers[image_idx],
      .renderArea =
          {
              .offset = {0, 0},
              .extent = _vk.swapchain.extent,
          },
      .clearValueCount = 2,
      .pClearValues    = clear_values,
  };

  vkCmdBeginRenderPass(cmd->cmd_buff, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
}

void umbvk_cmd_render_pass_end(umbvk_cmd_buffer* cmd) {
  vkCmdEndRenderPass(cmd->cmd_buff);
}

void umbvk_cmd_viewport(umbvk_cmd_buffer* cmd, f32 x, f32 y, f32 width, f32 height) {
  VkViewport viewport {
      viewport.x        = x,
      viewport.y        = y,
      viewport.width    = UMB_CLAMP_BOT(width, 1),
      viewport.height   = UMB_CLAMP_BOT(height, 1),
      viewport.minDepth = 0.0f,
      viewport.maxDepth = 1.0f,
  };
  vkCmdSetViewport(cmd->cmd_buff, 0, 1, &viewport);
}

void umbvk_cmd_scissor(umbvk_cmd_buffer* cmd, i32 x, i32 y, u32 width, u32 height) {
  VkRect2D scissor {
      .offset = {x, y},
      .extent = {.width = width, .height = height},
  };
  vkCmdSetScissor(cmd->cmd_buff, 0, 1, &scissor);
}

void umbvk_cmd_draw(
    umbvk_cmd_buffer* cmd,
    bool              indexed,
    u32               n_elts,
    u32               n_instances,
    u32               first_elt) {
  if (indexed) {
    vkCmdDrawIndexed(cmd->cmd_buff, n_elts, n_instances, first_elt, 0u, 0u);
  } else {
    vkCmdDraw(cmd->cmd_buff, n_elts, n_instances, first_elt, 0u);
  }
}

void umbvk_cmd_end(umbvk_cmd_buffer* cmd) {
  VK_CHECK(vkEndCommandBuffer(cmd->cmd_buff), "Failed to record command buffer!");
}

void umbvk_cmd_immediate(std::function<void(VkCommandBuffer cmd)>&& cmd_func) {
  VkCommandBuffer cmd = _vk.upload_context.cmd_buff;

  VkCommandBufferBeginInfo begin_info = {
      .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      .pInheritanceInfo = nullptr,
  };

  VK_CHECK(vkBeginCommandBuffer(cmd, &begin_info), "could not begin immediate command recording!");
  cmd_func(cmd);
  VK_CHECK(vkEndCommandBuffer(cmd), "could not end immediate buffer!");

  VkSubmitInfo submit_info {
      .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers    = &cmd,
  };

  VK_CHECK(
      vkQueueSubmit(_vk.graphics_queue, 1, &submit_info, _vk.upload_context.upload_fence),
      "failed to submit immediate command buffer!");

  vkWaitForFences(_vk.device, 1, &_vk.upload_context.upload_fence, VK_TRUE, UINT64_MAX);
  vkResetFences(_vk.device, 1, &_vk.upload_context.upload_fence);
  vkResetCommandPool(_vk.device, _vk.upload_context.cmd_pool, 0);
}

void umbvk_create_frame_resources() {
  VkSemaphoreCreateInfo semaphore_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
  };
  VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
  };

  for (i32 i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    VK_CHECK(
        vkCreateSemaphore(
            _vk.device,
            &semaphore_info,
            nullptr,
            &_vk.frames[i].image_available_semaphore),
        "failed to create semaphore");
    _vk.deletion_queue.push([=]() {
      vkDestroySemaphore(_vk.device, _vk.frames[i].image_available_semaphore, nullptr);
    });

    VK_CHECK(
        vkCreateSemaphore(
            _vk.device,
            &semaphore_info,
            nullptr,
            &_vk.frames[i].render_finished_semaphore),
        "failed to create semaphore");
    _vk.deletion_queue.push([=]() {
      vkDestroySemaphore(_vk.device, _vk.frames[i].render_finished_semaphore, nullptr);
    });

    VK_CHECK(
        vkCreateFence(_vk.device, &fence_info, nullptr, &_vk.frames[i].render_fence),
        "failed to create fence");
    _vk.deletion_queue.push(
        [=]() { vkDestroyFence(_vk.device, _vk.frames[i].render_fence, nullptr); });

    _vk.frames[i].cmd = umbvk_cmd_buffer_create();
  }

  VkFenceCreateInfo upload_fence_info {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
  };
  VK_CHECK(
      vkCreateFence(_vk.device, &upload_fence_info, nullptr, &_vk.upload_context.upload_fence),
      "failed to create upload fence!");

  _vk.deletion_queue.push(
      [=]() { vkDestroyFence(_vk.device, _vk.upload_context.upload_fence, nullptr); });
}

void umbvk_cmd_bind_gfx_descriptor_sets(
    umbvk_cmd_buffer* cmd,
    u32               set,
    VkDescriptorSet*  descriptor) {
  vkCmdBindDescriptorSets(
      cmd->cmd_buff,
      VK_PIPELINE_BIND_POINT_GRAPHICS,
      cmd->active_gfx_pipeline->pipeline_layout,
      set,
      1,
      descriptor,
      0,
      nullptr);
}

void umbvk_cmd_bind_gfx_descriptor_sets_offset(
    umbvk_cmd_buffer* cmd,
    u32               set,
    VkDescriptorSet*  descriptor,
    u32*              uniform_offset) {
  vkCmdBindDescriptorSets(
      cmd->cmd_buff,
      VK_PIPELINE_BIND_POINT_GRAPHICS,
      cmd->active_gfx_pipeline->pipeline_layout,
      set,
      1,
      descriptor,
      1,
      uniform_offset);
}

void umbvk_cmd_draw_objects(umbvk_cmd_buffer* cmd) {
  glm::mat4 model      = glm::translate(glm::mat4(1), glm::vec3(0, 5, 0));
  glm::vec3 camPos     = {0.f, -6.f, -10.f};
  glm::mat4 view       = glm::translate(glm::mat4(1.f), camPos);
  glm::mat4 projection = glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.0f);
  projection[1][1] *= -1;

  umb_gpu_camera_data cam_data = {
      .view     = view,
      .proj     = projection,
      .viewproj = projection * view,
  };

  void* data;
  vmaMapMemory(_vk.allocator, _vk.frames[_vk.frame_id].camera_buffer.alloc, &data);
  memcpy(data, &cam_data, sizeof(umb_gpu_camera_data));
  vmaUnmapMemory(_vk.allocator, _vk.frames[_vk.frame_id].camera_buffer.alloc);

  _vk.scene_parameters.ambient_color = {1.0f, 0.0f, 0.0f, 1.0f};
  byte* scene_data;
  vmaMapMemory(_vk.allocator, _vk.scene_parameters_buffer.alloc, (void**)&scene_data);
  scene_data += umbvk_pad_uniform_buffer_size(sizeof(umb_gpu_scene_data)) * _vk.frame_id;
  memcpy(scene_data, &_vk.scene_parameters, sizeof(umb_gpu_scene_data));
  vmaUnmapMemory(_vk.allocator, _vk.scene_parameters_buffer.alloc);

  void* obj_data;
  vmaMapMemory(_vk.allocator, _vk.frames[_vk.frame_id].object_buffer.alloc, &obj_data);
  umb_gpu_object_data* obj_ssbo = (umb_gpu_object_data*)obj_data;
  for (i32 i = 0; i < _vk.render_objects.len; ++i) {
    umb_render_object* o     = _vk.render_objects.data[i];
    obj_ssbo[i].model_matrix = o->transform;
  }
  vmaUnmapMemory(_vk.allocator, _vk.frames[_vk.frame_id].object_buffer.alloc);

  umb_mesh      last_mesh     = NULL;
  umb_material* last_material = NULL;
  for (i32 i = 0; i < _vk.render_objects.len; ++i) {
    umb_render_object* o = _vk.render_objects.data[i];

    if (o->material != last_material) {
      umbvk_cmd_bind_graphics_pipeline(cmd, &o->material->pipeline);
      last_material = o->material;

      u32 uniform_offset = umbvk_pad_uniform_buffer_size(sizeof(umb_gpu_scene_data)) * _vk.frame_id;
      umbvk_cmd_bind_gfx_descriptor_sets_offset(
          cmd,
          0,
          &_vk.frames[_vk.frame_id].global_descriptor,
          &uniform_offset);
      umbvk_cmd_bind_gfx_descriptor_sets(cmd, 1, &_vk.frames[_vk.frame_id].object_descriptor);
    }

    umb_push_constants constants {.render_matrix = model};
    umbvk_cmd_push_constants(cmd, &constants, VK_SHADER_STAGE_VERTEX_BIT);

    if (o->mesh != last_mesh) {
      VkDeviceSize offset = 0;
      umbvk_cmd_bind_vertex_buffer(cmd, 0, 1, &o->mesh->vertex_buffer.buffer, &offset);
      last_mesh = o->mesh;
    }

    umbvk_cmd_draw(cmd, false, o->mesh->vertices.len, 1, 0);
  }
}

void umbvk_set_allocator() {
  VmaAllocatorCreateInfo allocator_info = {
      .physicalDevice = _vk.physical_device,
      .device         = _vk.device,
      .instance       = _vk.instance,
  };

  vmaCreateAllocator(&allocator_info, &_vk.allocator);
}

void umb_gfx_init(umb_window* window) {
  _vk.arena          = umb_arena_create(UMB_MEGABYTES(4));
  _vk.render_objects = UMB_PTR_ARRAY_CREATE(umb_render_object, &_vk.arena, 1024);
  _vk.materials      = umb_hash_table_create(&_vk.arena, DEFAULT_NUM_SLOTS);
  _vk.meshes         = umb_hash_table_create(&_vk.arena, DEFAULT_NUM_SLOTS);

  VkApplicationInfo app_info {
      .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName   = "Game",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName        = "[umbral]",
      .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion         = VK_API_VERSION_1_2};

  _vk.window                        = window;
  umb_array_str required_extensions = umbvk_get_required_extensions(_vk.window);

  VkInstanceCreateInfo vulkan_create_info {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app_info,
    .enabledExtensionCount   = required_extensions.len,
    .ppEnabledExtensionNames = required_extensions.data,
#if defined(__APPLE__)
    .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
#endif
  };

  VkDebugUtilsMessengerCreateInfoEXT debug_create_info {};
  if (_vk.validation_layers_enabled) {
    UMB_ASSERT(umbvk_check_validation_layer_support(VALIDATION_LAYERS));
    vulkan_create_info.enabledLayerCount   = UMB_ARRAY_COUNT(VALIDATION_LAYERS, str);
    vulkan_create_info.ppEnabledLayerNames = VALIDATION_LAYERS;
    debug_messenger_create_info(debug_create_info);
    vulkan_create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debug_create_info;
  } else {
    vulkan_create_info.enabledLayerCount = 0;
    vulkan_create_info.pNext             = nullptr;
  }

  VK_CHECK(
      vkCreateInstance(&vulkan_create_info, nullptr, &_vk.instance),
      "failed to create instance!");

  // debug messenger for validation layers
  if (_vk.validation_layers_enabled) {
    VK_CHECK(
        umbvk_create_debug_utils_messenger_ext(
            _vk.instance,
            &debug_create_info,
            nullptr,
            &_vk.debug_messenger),
        "failed to create debug messenger!");
  }

  _vk.initialized = true;

  UMB_ASSERT(SDL_Vulkan_CreateSurface((SDL_Window*)window->raw_handle, _vk.instance, &_vk.surface));

  // Device
  umbvk_set_physical_device();
  umbvk_set_logical_device();
  umbvk_set_allocator();

  // RenderPass and Swapchain
  umbvk_swapchain_support_details swapchain_support =
      umbvk_query_swapchain_support(_vk.surface, _vk.physical_device);
  VkSurfaceFormatKHR surface_format = umbvk_choose_swap_surface_format(swapchain_support.formats);
  VkPresentModeKHR   present_mode = umbvk_choose_swap_present_mode(swapchain_support.present_modes);
  VkExtent2D         extent = umbvk_choose_swap_extent(_vk.window, swapchain_support.capabilities);

  // TODO(bryson): change API to just 'set_XXX'
  _vk.depth_format           = VK_FORMAT_D32_SFLOAT;
  _vk.compatible_render_pass = umbvk_create_render_pass(surface_format.format);
  _vk.swapchain = umbvk_create_swapchain(swapchain_support, surface_format, present_mode, extent);
  _vk.upload_context = umbvk_upload_context_create();

  umbvk_set_descriptors();

  // default material
  umb_material* default_gfx_material = umb_arena_push(&_vk.arena, umb_material);
  default_gfx_material->pipeline     = umbvk_default_graphics_pipeline_create();
  umb_gfx_register_material("default", default_gfx_material);

  umbvk_create_frame_resources();
}

void umb_gfx_shutdown() {
  if (_vk.initialized) {
    vkDeviceWaitIdle(_vk.device);

    _vk.deletion_queue.flush();

    vmaDestroyAllocator(_vk.allocator);

    umbvk_destroy_swapchain(&_vk.swapchain);
    vkDestroySurfaceKHR(_vk.instance, _vk.surface, nullptr);
    vkDestroyDevice(_vk.device, nullptr);
    if (_vk.validation_layers_enabled) {
      auto destroy_func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          _vk.instance,
          "vkDestroyDebugUtilsMessengerEXT");
      if (destroy_func != nullptr) destroy_func(_vk.instance, _vk.debug_messenger, nullptr);
    }
    vkDestroyInstance(_vk.instance, nullptr);
  }
}

void umb_gfx_draw_object(umb_render_object* o) {
  UMB_ARRAY_PUSH(_vk.render_objects, o);
}

void umb_gfx_register_mesh(str name, umb_mesh mesh) {
  const u64    buffer_size    = mesh->vertices.len * sizeof(umb_mesh_vertex);
  umbvk_buffer staging_buffer = umbvk_buffer_create_staging(buffer_size);

  void* data;
  vmaMapMemory(_vk.allocator, staging_buffer.alloc, &data);
  memcpy(data, mesh->vertices.data, buffer_size);
  vmaUnmapMemory(_vk.allocator, staging_buffer.alloc);

  mesh->vertex_buffer =
      umbvk_buffer_create_transfer(buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

  umbvk_cmd_immediate([=](VkCommandBuffer cmd) {
    VkBufferCopy copy = {
        .dstOffset = 0,
        .srcOffset = 0,
        .size      = buffer_size,
    };
    vkCmdCopyBuffer(cmd, staging_buffer.buffer, mesh->vertex_buffer.buffer, 1, &copy);
  });

  umb_hash_table_insert(&_vk.meshes, name, (byte*)mesh);
}

void umb_gfx_register_material(str name, umb_material* mat) {
  umb_hash_table_insert(&_vk.materials, name, (byte*)mat);
}

umb_mesh umb_gfx_get_mesh(str name) {
  return (umb_mesh)umb_hash_table_get(&_vk.meshes, name);
}
umb_material* umb_gfx_get_material(str name) {
  return (umb_material*)umb_hash_table_get(&_vk.materials, name);
}

umb_mesh umb_mesh_create(u32 n_vertices) {
  umb_mesh mesh  = umb_arena_push(&_vk.arena, umb_mesh_t);
  mesh->vertices = UMB_ARRAY_CREATE(umb_mesh_vertex, &_vk.arena, n_vertices);
  return mesh;
}

void umb_mesh_push_vertex(umb_mesh mesh, umb_mesh_vertex vertex) {
  UMB_ARRAY_PUSH(mesh->vertices, vertex);
}

// TODO(bryson): roll your own .obj parser?
// Regretably we must include some C++
#include <vector>
umb_mesh umb_mesh_load_from_obj(str filename) {
  tinyobj::attrib_t attrib;

  std::vector<tinyobj::shape_t>    shapes;
  std::vector<tinyobj::material_t> materials;

  std::string warn;
  std::string err;

  tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename, nullptr);
  if (!warn.empty()) { printf("tinobjloader: %s", warn.c_str()); }
  if (!err.empty()) {
    UMBI_LOG_ERROR("tinobjloader: %s", err.c_str());
    return NULL;
  }

  const int fv = 3;

  u64 n_vertices = 0;
  for (u64 s = 0; s < shapes.size(); ++s) {
    n_vertices += fv * shapes[s].mesh.num_face_vertices.size();
  }

  umb_mesh mesh = umb_mesh_create(n_vertices);
  for (u64 s = 0; s < shapes.size(); ++s) {
    u64 index_offset = 0;
    for (u64 f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
      // 3 for triangle!
      for (u64 v = 0; v < fv; ++v) {
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

        // vertex position
        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

        // vertex normal
        tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
        tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
        tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

        // vertex uv
        tinyobj::real_t ux = attrib.texcoords[2 * idx.texcoord_index + 0];
        tinyobj::real_t uy = attrib.texcoords[2 * idx.texcoord_index + 1];

        // copy it into our vertex
        umb_mesh_vertex new_vert;
        new_vert.position.x = vx;
        new_vert.position.y = vy;
        new_vert.position.z = vz;

        new_vert.normal.x = nx;
        new_vert.normal.y = ny;
        new_vert.normal.z = nz;

        new_vert.uv.x = ux;
        new_vert.uv.y = 1 - uy;

        // we are setting the vertex color as the vertex normal. This is just for display purposes
        new_vert.color = new_vert.normal;

        umb_mesh_push_vertex(mesh, new_vert);
      }
      index_offset += fv;
    }
  }

  return mesh;
}

void umb_gfx_draw_frame() {
  umbvk_frame*      frame = &_vk.frames[_vk.frame_id];
  umbvk_cmd_buffer* cmd   = &frame->cmd;

  vkWaitForFences(_vk.device, 1, &frame->render_fence, VK_TRUE, UINT64_MAX);

  u32      image_index;
  VkResult result = vkAcquireNextImageKHR(
      _vk.device,
      _vk.swapchain.swapchain,
      UINT64_MAX,
      frame->image_available_semaphore,
      VK_NULL_HANDLE,
      &image_index);
  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    umbvk_recreate_swapchain();
    return;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    printf("failed to acquire swap chain image!");
    UMB_ASSERT(false);
  }

  vkResetFences(_vk.device, 1, &frame->render_fence);

  // Record Command Buffer
  vkResetCommandBuffer(cmd->cmd_buff, 0);

  umbvk_cmd_begin(cmd);
  umbvk_cmd_render_pass_begin(cmd, image_index);

  umbvk_cmd_viewport(cmd, 0, 0, _vk.swapchain.extent.width, _vk.swapchain.extent.height);
  umbvk_cmd_scissor(cmd, 0, 0, _vk.swapchain.extent.width, _vk.swapchain.extent.height);

  umbvk_cmd_draw_objects(cmd);

  umbvk_cmd_render_pass_end(cmd);
  umbvk_cmd_end(cmd);

  // Present
  VkSemaphore wait_semaphores[] = {
      frame->image_available_semaphore,
  };
  VkSemaphore signal_semaphores[] = {
      frame->render_finished_semaphore,
  };
  VkPipelineStageFlags wait_stages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
  };

  VkSubmitInfo submit_info {
      .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .waitSemaphoreCount   = 1,
      .pWaitSemaphores      = wait_semaphores,
      .pWaitDstStageMask    = wait_stages,
      .commandBufferCount   = 1,
      .pCommandBuffers      = &cmd->cmd_buff,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores    = signal_semaphores,
  };

  VK_CHECK(
      vkQueueSubmit(_vk.graphics_queue, 1, &submit_info, frame->render_fence),
      "failed to submit draw command buffer!");

  VkSwapchainKHR   swapchains[] = {_vk.swapchain.swapchain};
  VkPresentInfoKHR present_info {
      .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores    = signal_semaphores,
      .swapchainCount     = 1,
      .pSwapchains        = swapchains,
      .pImageIndices      = &image_index,
      .pResults           = nullptr,
  };

  result = vkQueuePresentKHR(_vk.present_queue, &present_info);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      _vk.framebuffer_resized) {
    _vk.framebuffer_resized = false;
    umbvk_recreate_swapchain();
  } else if (result != VK_SUCCESS) {
    printf("failed to acquire swap chain image!");
    UMB_ASSERT(false);
  }

  _vk.frame_id = (_vk.frame_id + 1) % MAX_FRAMES_IN_FLIGHT;
}

void umb_gfx_framebuffer_resized() {
  _vk.framebuffer_resized = true;
}

void umb_gfx_register_texture(str name, umb_image image) {
  umb_texture tex = umb_arena_push(&_vk.arena, umb_texture_t);

  VkImageViewCreateInfo image_info {
      .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image    = image->image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format   = VK_FORMAT_R8G8B8A8_SRGB,
      .components =
          {
              .r = VK_COMPONENT_SWIZZLE_IDENTITY,
              .g = VK_COMPONENT_SWIZZLE_IDENTITY,
              .b = VK_COMPONENT_SWIZZLE_IDENTITY,
              .a = VK_COMPONENT_SWIZZLE_IDENTITY,
          },
      .subresourceRange =
          {
              .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
              .baseMipLevel   = 0,
              .levelCount     = 1,
              .baseArrayLayer = 0,
              .layerCount     = 1,
          },
  };

  vkCreateImageView(_vk.device, &image_info, nullptr, &tex->image_view);
  umb_hash_table_insert(&_vk.textures, name, (byte*)tex);
}

b32 umb_gfx_load_image_from_file(str file, umb_image out_image) {
  int      tex_width, tex_height, tex_channels;
  stbi_uc* pixels = stbi_load(file, &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
  if (!pixels) {
    printf("Failed to load texture from file: %s\n", file);
    return false;
  }

  void*        pixel_ptr  = pixels;
  VkDeviceSize image_size = tex_width * tex_height * 4;

  VkFormat image_format = VK_FORMAT_R8G8B8A8_SRGB;

  umbvk_buffer staging_buffer = umbvk_buffer_create_staging(image_size);

  void* data;
  vmaMapMemory(_vk.allocator, staging_buffer.alloc, &data);
  memcpy(data, pixel_ptr, (u64)image_size);
  vmaUnmapMemory(_vk.allocator, staging_buffer.alloc);

  stbi_image_free(pixels);

  VkExtent3D image_extent = {
      .width  = (u32)tex_width,
      .height = (u32)tex_height,
      .depth  = 1,
  };

  VkImageCreateInfo dimg_info = {
      .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType   = VK_IMAGE_TYPE_2D,
      .format      = image_format,
      .extent      = image_extent,
      .mipLevels   = 1,
      .arrayLayers = 1,
      .samples     = VK_SAMPLE_COUNT_1_BIT,
      .tiling      = VK_IMAGE_TILING_OPTIMAL,
      .usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
  };

  VmaAllocationCreateInfo dimg_allocinfo = {
      .usage = VMA_MEMORY_USAGE_GPU_ONLY,
  };

  umb_image_t image;
  vmaCreateImage(
      _vk.allocator,
      &dimg_info,
      &dimg_allocinfo,
      &image.image,
      &image.allocation,
      nullptr);

  umbvk_cmd_immediate([&](VkCommandBuffer cmd) {
    VkImageSubresourceRange range = {
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 0,
        .levelCount     = 1,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };

    VkImageMemoryBarrier image_barrier_to_transfer = {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .image            = image.image,
        .subresourceRange = range,
        .srcAccessMask    = 0,
        .dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
    };

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &image_barrier_to_transfer);

    VkBufferImageCopy copy_region = {
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel       = 0,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
        .imageExtent = image_extent,
    };

    vkCmdCopyBufferToImage(
        cmd,
        staging_buffer.buffer,
        image.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &copy_region);

    VkImageMemoryBarrier image_barrier_to_readable = {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .image            = image.image,
        .subresourceRange = range,
        .srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
    };

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &image_barrier_to_readable);
  });

  _vk.deletion_queue.push([=]() { vmaDestroyImage(_vk.allocator, image.image, image.allocation); });

  *out_image = image;

  return true;
}

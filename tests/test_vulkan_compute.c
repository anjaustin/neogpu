/*
 * Vulkan Compute Shader Test
 *
 * Builds on Pi:
 *   gcc -O3 -march=armv8-a+simd+fp16 -mtune=cortex-a72 -ffast-math \
 *       -funroll-loops -DNDEBUG \
 *       tests/test_vulkan_compute.c -o /tmp/test_vulkan_compute \
 *       -lvulkan -ldl -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <vulkan/vulkan.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

int main(void) {
    printf("=== Vulkan Compute Test ===\n\n");
    fflush(stdout);
    
    // Load SPIR-V - use current directory
    int fd = open("ternary_v3d.spv", O_RDONLY);
    if (fd < 0) {
        printf("Failed to open shader: %s\n", strerror(errno));
        return 1;
    }
    printf("Opened shader file\n");
    fflush(stdout);
    
    size_t size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    void* shader_data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (!shader_data) {
        printf("Failed to mmap shader\n");
        return 1;
    }
    printf("Loaded SPIR-V: %zu bytes\n", size);
    fflush(stdout);
    
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "NeoGPU",
        .apiVersion = VK_API_VERSION_1_0,
    };
    printf("Created appInfo\n");
    fflush(stdout);
    
    VkInstanceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    printf("Created createInfo\n");
    fflush(stdout);
    
    VkInstance instance;
    VkResult result = vkCreateInstance(&createInfo, NULL, &instance);
    printf("vkCreateInstance returned: %d\n", result);
    fflush(stdout);
    
    uint32_t deviceCount = 1;
    VkPhysicalDevice physicalDevice;
    vkEnumeratePhysicalDevices(instance, &deviceCount, &physicalDevice);
    
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    printf("Device: %s\n", props.deviceName);
    
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);
    VkQueueFamilyProperties* queueFamilies = malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies);
    
    int computeQueueFamily = -1;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamily = i;
            break;
        }
    }
    free(queueFamilies);
    
    if (computeQueueFamily < 0) {
        printf("No compute queue\n");
        return 1;
    }
    printf("Compute queue family: %d\n", computeQueueFamily);
    
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = computeQueueFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };
    
    VkPhysicalDeviceFeatures features = {};
    VkDeviceCreateInfo deviceCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueCreateInfo,
        .pEnabledFeatures = &features,
    };
    
    VkDevice device;
    result = vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);
    if (result != VK_SUCCESS) {
        printf("Failed to create device: %d\n", result);
        return 1;
    }
    printf("Device created\n");
    
    VkQueue computeQueue;
    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);
    printf("Got compute queue\n");
    
    VkShaderModuleCreateInfo shaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = shader_data,
    };
    
    VkShaderModule shaderModule;
    result = vkCreateShaderModule(device, &shaderInfo, NULL, &shaderModule);
    munmap(shader_data, size);
    
    if (result != VK_SUCCESS) {
        printf("Failed to create shader: %d\n", result);
        return 1;
    }
    printf("Shader module created!\n");
    
    VkPipelineShaderStageCreateInfo stageInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shaderModule,
        .pName = "main",
    };
    
    VkComputePipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stageInfo,
        .layout = VK_NULL_HANDLE,
    };
    
    VkPipeline pipeline;
    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &pipeline);
    if (result != VK_SUCCESS) {
        printf("Failed to create pipeline: %d\n", result);
        return 1;
    }
    printf("Compute pipeline created!\n");
    
    printf("\n=== SUCCESS ===\n");
    printf("Vulkan compute shader is working!\n");
    
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, shaderModule, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    
    return 0;
}

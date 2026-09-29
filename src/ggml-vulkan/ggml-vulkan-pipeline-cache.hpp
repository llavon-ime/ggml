#pragma once

// Vulkan-Hpp and its dispatcher are declared by ggml-vulkan.cpp before this
// internal header is included.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ggml::vulkan {

class PipelineCache final {
    using Checksum = std::uint64_t;
    static constexpr std::uint32_t maximum_bytes = 128u * 1024u * 1024u;

    struct Header {
        std::array<char, 8> magic{'G','G','M','L','V','K','C','1'};
        std::uint32_t version = 1;
        std::uint32_t vendor = 0;
        std::uint32_t device = 0;
        std::uint32_t driver = 0;
        std::uint32_t pointer_bytes = sizeof(void*);
        std::uint32_t size = 0;
        std::array<std::uint8_t, VK_UUID_SIZE> uuid{};
        Checksum checksum = 0;
    };
    static_assert(sizeof(Header) == 56);

    vk::Device device_;
    vk::PipelineCache cache_;
    Header identity_;
    std::filesystem::path path_;
    std::mutex mutex_;
    std::optional<Checksum> saved_checksum_;
    bool dirty_ = false;
    bool pipeline_created_ = false;

    static Checksum checksum(std::span<const std::uint8_t> data) noexcept {
        // Detect accidental corruption before passing opaque data to a driver.
        // This is not intended to authenticate user-writable cache files.
        constexpr Checksum offset_basis = 14695981039346656037ull;
        constexpr Checksum prime = 1099511628211ull;
        Checksum result = offset_basis;
        for (const auto byte : data) {
            result ^= byte;
            result *= prime;
        }
        return result;
    }

    static std::filesystem::path path_from_utf8(std::string_view value) {
        std::u8string converted(value.size(), u8'\0');
        std::memcpy(converted.data(), value.data(), value.size());
        return std::filesystem::path(converted);
    }

    bool valid_vulkan_header(std::span<const std::uint8_t> data) const noexcept {
        if (data.size() < 32) return false;
        std::array<std::uint32_t, 4> fields{};
        std::memcpy(fields.data(), data.data(), sizeof(fields));
        return fields[0] == 32 && fields[1] == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
               fields[2] == identity_.vendor && fields[3] == identity_.device &&
               std::equal(identity_.uuid.begin(), identity_.uuid.end(), data.begin() + 16);
    }

    std::vector<std::uint8_t> load() {
        std::ifstream file(path_, std::ios::binary | std::ios::ate);
        if (!file) return {};
        const auto file_size = file.tellg();
        if (file_size < static_cast<std::streamoff>(sizeof(Header)) ||
            file_size > static_cast<std::streamoff>(maximum_bytes + sizeof(Header))) return {};
        file.seekg(0);
        Header header{};
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!file || header.magic != identity_.magic || header.version != identity_.version ||
            header.vendor != identity_.vendor || header.device != identity_.device ||
            header.driver != identity_.driver || header.pointer_bytes != identity_.pointer_bytes ||
            header.uuid != identity_.uuid || header.size < 32 || header.size > maximum_bytes ||
            file_size != static_cast<std::streamoff>(sizeof(Header) + header.size)) return {};
        std::vector<std::uint8_t> data(header.size);
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        const auto value = checksum(data);
        if (!file || !valid_vulkan_header(data) || value != header.checksum) return {};
        saved_checksum_ = value;
        return data;
    }

public:
    PipelineCache() = default;
    PipelineCache(const PipelineCache &) = delete;
    PipelineCache & operator=(const PipelineCache &) = delete;

    void initialize(vk::Device device, const vk::PhysicalDeviceProperties & properties) noexcept {
        const std::lock_guard lock(mutex_);
        device_ = device;
        identity_.vendor = properties.vendorID;
        identity_.device = properties.deviceID;
        identity_.driver = properties.driverVersion;
        std::copy(properties.pipelineCacheUUID.begin(), properties.pipelineCacheUUID.end(),
                  identity_.uuid.begin());
    }

    // The host must opt in with an explicit UTF-8 directory before any compute
    // pipeline is created. No platform default or environment setting is used.
    bool configure(std::string_view directory) noexcept {
        const std::lock_guard lock(mutex_);
        try {
            if (!device_ || directory.empty() || pipeline_created_) return false;

            std::string uuid;
            for (const auto byte : identity_.uuid) uuid += std::format("{:02x}", byte);
            const auto requested_path = path_from_utf8(directory) / std::format(
                "ggml-0.11.1-v1-{:x}-{:x}-{:x}-{}-{}.bin",
                identity_.vendor, identity_.device, identity_.driver, sizeof(void *), uuid);
            if (cache_ && requested_path == path_) return true;

            path_ = requested_path;
            saved_checksum_.reset();
            auto data = load();
            vk::PipelineCache replacement;
            try {
                replacement = device_.createPipelineCache(
                    vk::PipelineCacheCreateInfo({}, data.size(), data.data()));
            } catch (const vk::SystemError &) {
                // A driver can reject otherwise well-formed, obsolete data.
                data.clear();
                saved_checksum_.reset();
                replacement = device_.createPipelineCache({});
            }
            if (cache_) device_.destroyPipelineCache(cache_);
            cache_ = replacement;
            dirty_ = false;
            std::fprintf(stderr, "[VK_CACHE] %s bytes=%zu\n",
                         data.empty() ? "miss" : "loaded", data.size());
            return true;
        } catch (...) {
            path_.clear();
            saved_checksum_.reset();
            std::fprintf(stderr, "[VK_CACHE] unavailable; continuing without disk reuse\n");
            return false;
        }
    }

    vk::Pipeline create(const vk::ComputePipelineCreateInfo & info) {
        const std::lock_guard lock(mutex_);
        auto pipeline = device_.createComputePipeline(cache_, info).value;
        pipeline_created_ = true;
        dirty_ = static_cast<bool>(cache_);
        return pipeline;
    }

    void save() noexcept {
        try {
            const std::lock_guard lock(mutex_);
            if (!cache_ || path_.empty() || !dirty_) return;
            std::size_t size = 0;
            if (device_.getPipelineCacheData(cache_, &size, nullptr) != vk::Result::eSuccess ||
                size < 32 || size > maximum_bytes) return;
            std::vector<std::uint8_t> data(size);
            if (device_.getPipelineCacheData(cache_, &size, data.data()) != vk::Result::eSuccess ||
                size > data.size()) return;
            data.resize(size);
            if (!valid_vulkan_header(data)) return;
            const auto value = checksum(data);
            if (saved_checksum_ == value) { dirty_ = false; return; }

            Header header = identity_;
            header.size = static_cast<std::uint32_t>(size);
            header.checksum = value;
            std::filesystem::create_directories(path_.parent_path());

            static std::atomic_uint64_t sequence{};
            const auto clock = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            std::filesystem::path temporary;
            bool written = false;
            for (unsigned int attempt = 0; attempt < 16 && !written; ++attempt) {
                temporary = path_;
                temporary += std::format(".{:x}.{:x}.tmp", clock, sequence.fetch_add(1));
                std::ofstream file(temporary,
                    std::ios::binary | std::ios::out | std::ios::noreplace);
                if (!file) continue;
                file.write(reinterpret_cast<const char *>(&header), sizeof(header));
                file.write(reinterpret_cast<const char *>(data.data()),
                           static_cast<std::streamsize>(data.size()));
                file.close();
                if (!file) {
                    std::error_code ignored;
                    std::filesystem::remove(temporary, ignored);
                    throw std::runtime_error("pipeline cache write failed");
                }
                written = true;
            }
            if (!written) throw std::runtime_error("pipeline cache temporary file creation failed");

            try {
                std::filesystem::rename(temporary, path_);
            } catch (...) {
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                throw;
            }
            saved_checksum_ = value;
            dirty_ = false;
            std::fprintf(stderr, "[VK_CACHE] saved bytes=%zu\n", size);
        } catch (...) {
            std::fprintf(stderr, "[VK_CACHE] save failed; inference remains available\n");
        }
    }

    void close() noexcept {
        save();
        const std::lock_guard lock(mutex_);
        if (cache_) device_.destroyPipelineCache(cache_);
        cache_ = nullptr;
    }
};

} // namespace ggml::vulkan

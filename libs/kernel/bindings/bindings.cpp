// #include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/utils/init_helper.hpp>
#include <iostream>
#include <kernel/add.hpp>
#include <span>
#include <unordered_map>
#include <utils/to_span.hpp>

namespace py = pybind11;

namespace {

struct DeviceBufferHandle {
    uint64_t id;
};

class DeviceBufferRegistry {
public:
    auto create(gpu::DeviceBuffer&& buf) -> DeviceBufferHandle {
        uint64_t const id{++m_nextId};
        m_buffers.emplace(id, std::move(buf));
        return DeviceBufferHandle{id};
    }

    auto get(DeviceBufferHandle h) -> gpu::DeviceBuffer& { return m_buffers.at(h.id); }
    auto remove(DeviceBufferHandle h) -> void { m_buffers.erase(h.id); }

private:
    uint64_t m_nextId{0};
    std::unordered_map<uint64_t, gpu::DeviceBuffer> m_buffers{};
};

static DeviceBufferRegistry bufferRegistry;

}  // namespace

PYBIND11_MODULE(python_kernel, m) {
    m.doc() = "Kernel C++ bindings";

    py::class_<DeviceBufferHandle>(m, "DeviceBufferHandle").def_readonly("id", &DeviceBufferHandle::id);

    m.def(
        "create_device_buffer",
        [](py::buffer data) -> DeviceBufferHandle {
            auto buf{data.request()};

            if (buf.ndim != 1) {
                throw std::runtime_error("Expected 1D array");
            }

            gpu::DeviceBuffer deviceBuffer{};
            if (auto res{deviceBuffer.init(buf.size * buf.itemsize)}; !res) {
                throw std::runtime_error(res.error());
            }

            if (auto res{gpu::utils::init_buffer_sync(deviceBuffer,  //
                                                      utils::to_byte_span(static_cast<char const*>(buf.ptr),  //
                                                                          buf.size * buf.itemsize))};
                !res) {
                throw std::runtime_error(res.error());
            }

            return bufferRegistry.create(std::move(deviceBuffer));
        },
        py::arg("data"));

    m.def(
        "create_device_buffer",
        [](size_t size) -> DeviceBufferHandle {
            gpu::DeviceBuffer deviceBuffer{};
            if (auto res{deviceBuffer.init(size)}; !res) {
                throw std::runtime_error(res.error());
            }

            return bufferRegistry.create(std::move(deviceBuffer));
        },
        py::arg("size"));

    m.def("init", []() {
        if (auto const res{gpu::GpuManager::init()}; !res) {
            throw std::runtime_error(res.error());
        }
    });

    m.def("destroy_device_buffer", [](DeviceBufferHandle h) {
        gpu::DeviceBuffer& buffer{bufferRegistry.get(h)};
        buffer.destroy();
        bufferRegistry.remove(h);
    });

    m.def("clear", []() { gpu::GpuManager::destroy(); });

    m.def("add", [](DeviceBufferHandle a, DeviceBufferHandle b, DeviceBufferHandle result) {
        gpu::DeviceBuffer const& aDevice{bufferRegistry.get(a)};
        gpu::DeviceBuffer const& bDevice{bufferRegistry.get(b)};
        gpu::DeviceBuffer const& resultDevice{bufferRegistry.get(result)};

        if (auto const res{kernel::add(aDevice, bDevice, resultDevice)}; !res) {
            throw std::runtime_error(res.error());
        }
    });

    // m.def(
    //     "add",
    //     [](py::buffer a, py::buffer b, py::buffer r) {
    //         using Clock = std::chrono::high_resolution_clock;
    //         using Duration = std::chrono::duration<double, std::micro>;

    //         auto const aBuf{a.request()};
    //         auto const bBuf{b.request()};
    //         auto const rBuf{r.request()};

    //         if (aBuf.ndim != 1 || bBuf.ndim != 1 || rBuf.ndim != 1) {
    //             throw std::runtime_error("expected 1D arrays");
    //         }

    //         {
    //             py::gil_scoped_release release;

    //             auto const* aPtr{static_cast<float const*>(aBuf.ptr)};
    //             auto const* bPtr{static_cast<float const*>(bBuf.ptr)};
    //             auto* rPtr{static_cast<float*>(rBuf.ptr)};

    //             // std::span<float const> aSpan(aPtr, aBuf.size);
    //             // std::span<float const> bSpan(bPtr, bBuf.size);
    //             // std::span<float> rSpan(rPtr, rBuf.size);

    //             std::vector<float> aHost(aBuf.size);
    //             std::vector<float> bHost(aBuf.size);
    //             std::vector<float> resultHost(aBuf.size);

    //             auto const start{Clock::now()};

    //             // if (auto const res{kernel::add(aSpan, bSpan, rSpan)}; !res) {
    //             //     throw std::runtime_error(res.error());
    //             // }
    //             if (auto const res{kernel::add(aHost, bHost, resultHost)}; !res) {
    //                 throw std::runtime_error(res.error());
    //             }

    //             std::cout << std::chrono::duration_cast<Duration>(Clock::now() - start).count() << " us" <<
    //             std::endl;
    //         }
    //     },
    //     py::arg("a"), py::arg("b"), py::arg("result"));
}
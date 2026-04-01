#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <rga/im2d.hpp>
#include <rga/rga.h>

namespace py = pybind11;

static std::atomic<unsigned long long> g_rga_call_seq{0};

static bool rga_debug_enabled() {
    const char *val = std::getenv("ROCKCHIP_RGA_DEBUG");
    if (val == nullptr || val[0] == '\0') {
        return false;
    }
    if (val[0] == '0' && val[1] == '\0') {
        return false;
    }
    return true;
}

static void rga_debug_log(const char *fmt, ...) {
    if (!rga_debug_enabled()) {
        return;
    }
    std::fprintf(stderr, "[rockchip_rga] ");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

static void ensure_bgr_u8_c(const py::array &arr, const char *name) {
    if (arr.ndim() != 3) {
        throw std::runtime_error(std::string(name) + " must be HxWx3");
    }
    if (arr.shape(2) != 3) {
        throw std::runtime_error(std::string(name) + " channel must be 3 (BGR)");
    }
    if (arr.itemsize() != 1) {
        throw std::runtime_error(std::string(name) + " dtype must be uint8");
    }
    if (!(arr.flags() & py::array::c_style)) {
        throw std::runtime_error(std::string(name) + " must be contiguous C-order array");
    }
}

static void ensure_gray_u8_c(const py::array &arr, const char *name) {
    if (arr.ndim() != 2) {
        throw std::runtime_error(std::string(name) + " must be HxW");
    }
    if (arr.itemsize() != 1) {
        throw std::runtime_error(std::string(name) + " dtype must be uint8");
    }
    if (!(arr.flags() & py::array::c_style)) {
        throw std::runtime_error(std::string(name) + " must be contiguous C-order array");
    }
}

static py::tuple preprocess_pair_bgr_to_gray(
    py::array left_bgr,
    py::array right_bgr,
    double scale
) {
    const unsigned long long seq = ++g_rga_call_seq;
    rga_debug_log(
        "seq=%llu enter preprocess_pair_bgr_to_gray scale=%.6f left_ndim=%lld right_ndim=%lld",
        seq,
        scale,
        static_cast<long long>(left_bgr.ndim()),
        static_cast<long long>(right_bgr.ndim())
    );

    if (scale <= 0.0 || scale > 1.0) {
        rga_debug_log("seq=%llu invalid scale=%.6f", seq, scale);
        throw std::runtime_error("scale must be in (0, 1]");
    }

    try {
        ensure_bgr_u8_c(left_bgr, "left_bgr");
        ensure_bgr_u8_c(right_bgr, "right_bgr");
    } catch (const std::exception &exc) {
        rga_debug_log("seq=%llu input validation failed: %s", seq, exc.what());
        throw;
    }

    const int h = static_cast<int>(left_bgr.shape(0));
    const int w = static_cast<int>(left_bgr.shape(1));
    rga_debug_log("seq=%llu validated BGR input size=%dx%d", seq, w, h);

    if (right_bgr.shape(0) != left_bgr.shape(0) || right_bgr.shape(1) != left_bgr.shape(1)) {
        rga_debug_log("seq=%llu shape mismatch between left_bgr and right_bgr", seq);
        throw std::runtime_error("left_bgr and right_bgr must have the same shape");
    }

    const int nw = std::max(1, static_cast<int>(w * scale));
    const int nh = std::max(1, static_cast<int>(h * scale));
    rga_debug_log("seq=%llu target size=%dx%d", seq, nw, nh);

    auto left_resized = py::array_t<uint8_t>({nh, nw, 3});
    auto right_resized = py::array_t<uint8_t>({nh, nw, 3});
    auto left_gray = py::array_t<uint8_t>({nh, nw});
    auto right_gray = py::array_t<uint8_t>({nh, nw});

    auto *left_in_ptr = static_cast<uint8_t *>(left_bgr.mutable_data());
    auto *right_in_ptr = static_cast<uint8_t *>(right_bgr.mutable_data());

    auto *left_resized_ptr = static_cast<uint8_t *>(left_resized.mutable_data());
    auto *right_resized_ptr = static_cast<uint8_t *>(right_resized.mutable_data());
    auto *left_gray_ptr = static_cast<uint8_t *>(left_gray.mutable_data());
    auto *right_gray_ptr = static_cast<uint8_t *>(right_gray.mutable_data());

    rga_debug_log(
        "seq=%llu ptrs left_in=%p right_in=%p left_resized=%p right_resized=%p left_gray=%p right_gray=%p",
        seq,
        left_in_ptr,
        right_in_ptr,
        left_resized_ptr,
        right_resized_ptr,
        left_gray_ptr,
        right_gray_ptr
    );

    rga_debug_log("seq=%llu wrapbuffer start", seq);
    rga_buffer_t src_l = wrapbuffer_virtualaddr(left_in_ptr, w, h, RK_FORMAT_BGR_888, w, h);
    rga_buffer_t src_r = wrapbuffer_virtualaddr(right_in_ptr, w, h, RK_FORMAT_BGR_888, w, h);
    rga_buffer_t dst_l = wrapbuffer_virtualaddr(left_resized_ptr, nw, nh, RK_FORMAT_BGR_888, nw, nh);
    rga_buffer_t dst_r = wrapbuffer_virtualaddr(right_resized_ptr, nw, nh, RK_FORMAT_BGR_888, nw, nh);
    rga_buffer_t gray_l = wrapbuffer_virtualaddr(left_gray_ptr, nw, nh, RK_FORMAT_YCbCr_400, nw, nh);
    rga_buffer_t gray_r = wrapbuffer_virtualaddr(right_gray_ptr, nw, nh, RK_FORMAT_YCbCr_400, nw, nh);
    rga_debug_log("seq=%llu wrapbuffer done", seq);

    rga_debug_log("seq=%llu imresize(left) start", seq);
    IM_STATUS st = imresize(src_l, dst_l);
    rga_debug_log("seq=%llu imresize(left) status=%d", seq, static_cast<int>(st));
    if (st != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imresize failed (left)");
    }

    rga_debug_log("seq=%llu imresize(right) start", seq);
    st = imresize(src_r, dst_r);
    rga_debug_log("seq=%llu imresize(right) status=%d", seq, static_cast<int>(st));
    if (st != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imresize failed (right)");
    }

    rga_debug_log("seq=%llu imcvtcolor(left) start", seq);
    st = imcvtcolor(dst_l, gray_l, RK_FORMAT_BGR_888, RK_FORMAT_YCbCr_400);
    rga_debug_log("seq=%llu imcvtcolor(left) status=%d", seq, static_cast<int>(st));
    if (st != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imcvtcolor failed (left)");
    }

    rga_debug_log("seq=%llu imcvtcolor(right) start", seq);
    st = imcvtcolor(dst_r, gray_r, RK_FORMAT_BGR_888, RK_FORMAT_YCbCr_400);
    rga_debug_log("seq=%llu imcvtcolor(right) status=%d", seq, static_cast<int>(st));
    if (st != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imcvtcolor failed (right)");
    }

    rga_debug_log("seq=%llu preprocess_pair_bgr_to_gray done", seq);

    return py::make_tuple(left_resized, right_resized, left_gray, right_gray);
}

static py::tuple preprocess_pair_gray_to_gray(
    py::array left_gray,
    py::array right_gray,
    double scale
) {
    const unsigned long long seq = ++g_rga_call_seq;
    rga_debug_log(
        "seq=%llu enter preprocess_pair_gray_to_gray scale=%.6f left_ndim=%lld right_ndim=%lld",
        seq,
        scale,
        static_cast<long long>(left_gray.ndim()),
        static_cast<long long>(right_gray.ndim())
    );

    if (scale <= 0.0 || scale > 1.0) {
        rga_debug_log("seq=%llu invalid scale=%.6f", seq, scale);
        throw std::runtime_error("scale must be in (0, 1]");
    }

    try {
        ensure_gray_u8_c(left_gray, "left_gray");
        ensure_gray_u8_c(right_gray, "right_gray");
    } catch (const std::exception &exc) {
        rga_debug_log("seq=%llu input validation failed: %s", seq, exc.what());
        throw;
    }

    const int h = static_cast<int>(left_gray.shape(0));
    const int w = static_cast<int>(left_gray.shape(1));
    rga_debug_log("seq=%llu validated gray input size=%dx%d", seq, w, h);

    if (right_gray.shape(0) != left_gray.shape(0) || right_gray.shape(1) != left_gray.shape(1)) {
        rga_debug_log("seq=%llu shape mismatch between left_gray and right_gray", seq);
        throw std::runtime_error("left_gray and right_gray must have the same shape");
    }

    const int nw = std::max(1, static_cast<int>(w * scale));
    const int nh = std::max(1, static_cast<int>(h * scale));
    rga_debug_log("seq=%llu target size=%dx%d", seq, nw, nh);

    auto left_resized = py::array_t<uint8_t>({nh, nw});
    auto right_resized = py::array_t<uint8_t>({nh, nw});

    auto *left_in_ptr = static_cast<uint8_t *>(left_gray.mutable_data());
    auto *right_in_ptr = static_cast<uint8_t *>(right_gray.mutable_data());
    auto *left_resized_ptr = static_cast<uint8_t *>(left_resized.mutable_data());
    auto *right_resized_ptr = static_cast<uint8_t *>(right_resized.mutable_data());

    rga_debug_log(
        "seq=%llu ptrs left_in=%p right_in=%p left_resized=%p right_resized=%p",
        seq,
        left_in_ptr,
        right_in_ptr,
        left_resized_ptr,
        right_resized_ptr
    );

    rga_debug_log("seq=%llu wrapbuffer start", seq);
    rga_buffer_t src_l = wrapbuffer_virtualaddr(left_in_ptr, w, h, RK_FORMAT_YCbCr_400, w, h);
    rga_buffer_t src_r = wrapbuffer_virtualaddr(right_in_ptr, w, h, RK_FORMAT_YCbCr_400, w, h);
    rga_buffer_t dst_l = wrapbuffer_virtualaddr(left_resized_ptr, nw, nh, RK_FORMAT_YCbCr_400, nw, nh);
    rga_buffer_t dst_r = wrapbuffer_virtualaddr(right_resized_ptr, nw, nh, RK_FORMAT_YCbCr_400, nw, nh);
    rga_debug_log("seq=%llu wrapbuffer done", seq);

    rga_debug_log("seq=%llu imresize(left gray) start", seq);
    IM_STATUS st = imresize(src_l, dst_l);
    rga_debug_log("seq=%llu imresize(left gray) status=%d", seq, static_cast<int>(st));
    if (st != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imresize failed (left gray)");
    }

    rga_debug_log("seq=%llu imresize(right gray) start", seq);
    st = imresize(src_r, dst_r);
    rga_debug_log("seq=%llu imresize(right gray) status=%d", seq, static_cast<int>(st));
    if (st != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imresize failed (right gray)");
    }

    // For gray-direct mode, resized frames are already the matcher inputs.
    rga_debug_log("seq=%llu preprocess_pair_gray_to_gray done", seq);
    return py::make_tuple(left_resized, right_resized, left_resized, right_resized);
}

PYBIND11_MODULE(rockchip_rga, m) {
    m.doc() = "RK3588 RGA preprocess backend (BGR/GRAY resize + gray output)";
    rga_debug_log("module init (set ROCKCHIP_RGA_DEBUG=1 to keep native debug logs enabled)");
    m.def("is_available", []() {
        return true;
    });
    m.def("preprocess_pair_bgr_to_gray", &preprocess_pair_bgr_to_gray,
          py::arg("left_bgr"), py::arg("right_bgr"), py::arg("scale"));
    m.def("preprocess_pair_gray_to_gray", &preprocess_pair_gray_to_gray,
          py::arg("left_gray"), py::arg("right_gray"), py::arg("scale"));
}

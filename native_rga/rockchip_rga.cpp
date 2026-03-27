#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <rga/im2d.hpp>
#include <rga/rga.h>

namespace py = pybind11;

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

static py::tuple preprocess_pair_bgr_to_gray(
    py::array left_bgr,
    py::array right_bgr,
    double scale
) {
    if (scale <= 0.0 || scale > 1.0) {
        throw std::runtime_error("scale must be in (0, 1]");
    }

    ensure_bgr_u8_c(left_bgr, "left_bgr");
    ensure_bgr_u8_c(right_bgr, "right_bgr");

    const int h = static_cast<int>(left_bgr.shape(0));
    const int w = static_cast<int>(left_bgr.shape(1));
    if (right_bgr.shape(0) != left_bgr.shape(0) || right_bgr.shape(1) != left_bgr.shape(1)) {
        throw std::runtime_error("left_bgr and right_bgr must have the same shape");
    }

    const int nw = std::max(1, static_cast<int>(w * scale));
    const int nh = std::max(1, static_cast<int>(h * scale));

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

    rga_buffer_t src_l = wrapbuffer_virtualaddr(left_in_ptr, w, h, RK_FORMAT_BGR_888, w, h);
    rga_buffer_t src_r = wrapbuffer_virtualaddr(right_in_ptr, w, h, RK_FORMAT_BGR_888, w, h);
    rga_buffer_t dst_l = wrapbuffer_virtualaddr(left_resized_ptr, nw, nh, RK_FORMAT_BGR_888, nw, nh);
    rga_buffer_t dst_r = wrapbuffer_virtualaddr(right_resized_ptr, nw, nh, RK_FORMAT_BGR_888, nw, nh);
    rga_buffer_t gray_l = wrapbuffer_virtualaddr(left_gray_ptr, nw, nh, RK_FORMAT_YCbCr_400, nw, nh);
    rga_buffer_t gray_r = wrapbuffer_virtualaddr(right_gray_ptr, nw, nh, RK_FORMAT_YCbCr_400, nw, nh);

    IM_STATUS st = imresize(src_l, dst_l);
    if (st != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imresize failed (left)");
    }
    st = imresize(src_r, dst_r);
    if (st != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imresize failed (right)");
    }

    st = imcvtcolor(dst_l, gray_l, RK_FORMAT_BGR_888, RK_FORMAT_YCbCr_400);
    if (st != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imcvtcolor failed (left)");
    }
    st = imcvtcolor(dst_r, gray_r, RK_FORMAT_BGR_888, RK_FORMAT_YCbCr_400);
    if (st != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA imcvtcolor failed (right)");
    }

    return py::make_tuple(left_resized, right_resized, left_gray, right_gray);
}

PYBIND11_MODULE(rockchip_rga, m) {
    m.doc() = "RK3588 RGA preprocess backend (resize + BGR->GRAY)";
    m.def("is_available", []() {
        return true;
    });
    m.def("preprocess_pair_bgr_to_gray", &preprocess_pair_bgr_to_gray,
          py::arg("left_bgr"), py::arg("right_bgr"), py::arg("scale"));
}

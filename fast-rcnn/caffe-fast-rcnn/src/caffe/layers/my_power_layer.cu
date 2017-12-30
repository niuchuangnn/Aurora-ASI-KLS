#include <vector>

#include "caffe/layers/my_power_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>
using namespace std;

namespace caffe {
    template <typename Dtype>
    void MyPowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        const int count = top[0]->count();
        Dtype* top_data = top[0]->mutable_gpu_data();
        caffe_gpu_powx(count, bottom[0]->gpu_data(), Dtype(power_), top_data);
    }

    template <typename Dtype>
    void MyPowerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
        const int count = top[0]->count();
        const Dtype* top_diff = top[0]->gpu_diff();
        if (propagate_down[0]) {
            const Dtype* bottom_data = bottom[0]->gpu_data();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const Dtype* bottom_data_w = bottom[0]->cpu_data();
            const Dtype* bottom_diff_w = bottom[0]->cpu_diff();

            cout << "bottom_data[0]: " << bottom_data_w[0] << endl;
            cout << "bottom_diff[0]: " << bottom_diff_w[0] << endl;

            caffe_gpu_powx(count, bottom_data, Dtype(power_ - 1), bottom_diff);

            bottom_diff = bottom[0]->mutable_gpu_diff();
            bottom_data_w = bottom[0]->cpu_data();
            bottom_diff_w = bottom[0]->cpu_diff();
            cout << "bottom_data[0]: " << bottom_data_w[0] << endl;
            cout << "bottom_diff[0]: " << bottom_diff_w[0] << endl;

            caffe_gpu_scal(count, Dtype(power_), bottom_diff);

            bottom_diff = bottom[0]->mutable_gpu_diff();
            bottom_data_w = bottom[0]->cpu_data();
            bottom_diff_w = bottom[0]->cpu_diff();
            cout << "bottom_data[0]: " << bottom_data_w[0] << endl;
            cout << "bottom_diff[0]: " << bottom_diff_w[0] << endl;

            const Dtype* top_diff_w = top[0]->cpu_diff();
            cout << "top_diff[0]: " << top_diff_w[0] << endl;

            caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);

            bottom_diff = bottom[0]->mutable_gpu_diff();
            bottom_data_w = bottom[0]->cpu_data();
            bottom_diff_w = bottom[0]->cpu_diff();
            cout << "bottom_data[0]: " << bottom_data_w[0] << endl;
            cout << "bottom_diff[0]: " << bottom_diff_w[0] << endl;
        }
    }

    INSTANTIATE_LAYER_GPU_FUNCS(MyPowerLayer);


}  // namespace caffe

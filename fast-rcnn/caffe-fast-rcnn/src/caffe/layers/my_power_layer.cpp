//
// Created by niuchuang on 16-12-23.
//

#include <vector>

#include "caffe/layers/my_power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template <typename Dtype>
    void MyPowerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        NeuronLayer<Dtype>::LayerSetUp(bottom, top);
        CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
        power_ = this->layer_param_.my_power_param().power();
    }

    template <typename Dtype>
    void MyPowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        const int count = top[0]->count();
        Dtype* top_data = top[0]->mutable_cpu_data();
        caffe_powx(count, bottom[0]->cpu_data(), Dtype(power_), top_data);
    }

    template <typename Dtype>
    void MyPowerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        const int count = top[0]->count();
        const Dtype* top_diff = top[0]->cpu_diff();
        if (propagate_down[0]) {
            const Dtype* bottom_data = bottom[0]->cpu_data();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            caffe_powx(count, bottom_data, Dtype(power_ - 1), bottom_diff);
            caffe_scal(count, Dtype(power_), bottom_diff);
            caffe_mul(count, bottom_diff, top_diff, bottom_diff);
        }
    }

    #ifdef CPU_ONLY
    STUB_GPU(MyPower);
    #endif

    INSTANTIATE_CLASS(MyPowerLayer);
    REGISTER_LAYER_CLASS(MyPower);

}  // namespace caffe
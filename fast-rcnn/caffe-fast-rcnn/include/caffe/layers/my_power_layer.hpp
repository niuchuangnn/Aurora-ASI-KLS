//
// Created by niuchuang on 16-12-23.
//

#ifndef CAFFE_TEST_POWER_HPP
#define CAFFE_TEST_POWER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Computes @f$ y = x^n @f$
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the inputs @f$ x @f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the computed outputs @f$ y = x^2 @f$
 */
template <typename Dtype>
class MyPowerLayer : public NeuronLayer<Dtype> {
    public:
        explicit MyPowerLayer(const LayerParameter& param)
                : NeuronLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "testPower"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        /// @copydoc MyPower
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);

        /**
         * @brief Computes the error gradient w.r.t. the absolute value inputs.
         *
         * @param top output Blob vector (length 1), providing the error gradient with
         *      respect to the outputs
         *   -# @f$ (N \times C \times H \times W) @f$
         *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
         *      with respect to computed outputs @f$ y @f$
         * @param propagate_down see Layer::Backward.
         * @param bottom input Blob vector (length 2)
         *   -# @f$ (N \times C \times H \times W) @f$
         *      the inputs @f$ x @f$; Backward fills their diff with
         *      gradients @f$
         *        \frac{\partial E}{\partial x} =
         *            \mathrm{sign}(x) \frac{\partial E}{\partial y}
         *      @f$ if propagate_down[0]
         */
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        float power_;
    };

}  // namespace caffe

#endif //CAFFE_TEST_POWER_HPP

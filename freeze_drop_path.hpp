#ifndef CAFFE_FREEZE_DROP_PATH_HPP_
#define CAFFE_FREEZE_DROP_PATH_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * \ingroup ttic
 * @brief Only one path is active at a time during training.  Higher paths, if any, are
 *        frozen and lower paths are dropped.
 *        This is used with several network branches, where the lower branches
 *        learn a correction to the upper branches results.
 * 
 * @author Leslie N. Smith
 */
template <typename Dtype>
class FreezeDropPathLayer : public Layer<Dtype> {
 public:
  explicit FreezeDropPathLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FreezeDropPathLayer"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

//  std::vector<bool> drops_;
  unsigned int iteration, branch, lastBranch;
  bool stochastic;
  std::vector<Dtype> thresholds_;
  std::vector<unsigned int> uint_thresholds_;
};

}  // namespace caffe

#endif  // CAFFE_FREEZE_DROP_PATH_HPP_

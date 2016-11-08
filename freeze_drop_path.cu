#include <vector>

#include "caffe/layers/freeze_drop_path.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdlib.h>     /* srand, rand */

namespace caffe {

template <typename Dtype>
void FreezeDropPathLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->phase_ == TRAIN) {
	branch = 0;
	if (stochastic) {  
      uint rnd = std::rand() % uint_thresholds_[bottom.size()];
      for (int i = 0; i < bottom.size(); ++i) {
		  if (rnd >= uint_thresholds_[i] && rnd < uint_thresholds_[i+1]) branch = i;
      }
//	LOG(INFO) << " rnd = " << rnd <<  "branch " << branch;
    } else {
		iteration = (iteration + 1) % this->layer_param_.freeze_drop_path_param().num_iter_per_cycle();
        for (int i = 0; i < bottom.size(); ++i) {
		  if (iteration >= thresholds_[i] && iteration < thresholds_[i+1]) branch = i;
		}
		if (branch > lastBranch || branch < lastBranch) 
		  LOG(INFO) << " Changing from branch " << lastBranch << " to " << branch;
//	LOG(INFO) << " iteration  = " << iteration << " bottom.size = " << bottom.size() << "branch " << branch;
	}
  } else {
	  branch = bottom.size() - 1;
  }
  
//	Zero the top layer before adding the bottom data
  caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  for (int i = 0; i <= branch; ++i) {
      caffe_gpu_axpy(top[0]->count(), Dtype(1.0), bottom[i]->cpu_data(), top[0]->mutable_cpu_data());
  }
  lastBranch = branch;
}

template <typename Dtype>
void FreezeDropPathLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < bottom.size(); ++i) {
	  if (propagate_down[i]) {
        if (i == branch) {
          caffe_copy(bottom[i]->count(), top[0]->cpu_diff(), bottom[i]->mutable_cpu_diff());
        } else {
          caffe_gpu_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_cpu_diff());
        }
      }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(FreezeDropPathLayer);

}  // namespace caffe

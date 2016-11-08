#include <vector>

#include "caffe/layers/freeze_drop_path.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FreezeDropPathLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype num_iter_per_cycle = this->layer_param_.freeze_drop_path_param().num_iter_per_cycle();
  Dtype interval_type = this->layer_param_.freeze_drop_path_param().interval_type();

  thresholds_.clear();
  uint_thresholds_.clear();
  stochastic = false;
  if ( num_iter_per_cycle == 0) stochastic = true;
  iteration = 0;
  lastBranch = 0;

  Dtype interval[20];
  Dtype norm = 0.0;
  interval[0] = 0.0;
  Dtype x;
  if (interval_type == 0) {
	  for (int i = 1; i <= bottom.size(); ++i) {
		x = pow(Dtype(i),2.0);
		interval[i] += interval[i-1]  + x;
		norm += x;
	  }
  } else if(interval_type == 1) {
	  for (int i = 1; i <= bottom.size(); ++i) {
		interval[i] = i;
	  }
      norm = bottom.size();
  } else if(interval_type == 2) {
	  for (int i = 0; i <= bottom.size(); ++i) {
		x = pow(Dtype(bottom.size()-i),2.0);
		interval[i] += interval[i]  + x;
		norm += x;
	  }
  } else {
	  // Not recognized
	  LOG(ERROR) << " interval_type "  << interval_type << " not recognized. ";
  }
	  
	  
//  LOG(INFO) << " num_iter_per_cycle = " << num_iter_per_cycle << " bottom.size = " << bottom.size() << " norm = " << norm;
	
    
  if (stochastic) {  
	  for (int i = 0; i <= bottom.size(); ++i) {
        thresholds_.push_back(interval[i]/norm);
		DCHECK(thresholds_[i] >= 0.);
		DCHECK(thresholds_[i] <= 1.);
        uint_thresholds_.push_back(static_cast<unsigned int>(interval[i]));
	  }
  } else {
	  for (int i = 0; i <= bottom.size(); ++i) {
		x =  Dtype(num_iter_per_cycle * interval[i])/norm;
		thresholds_.push_back(x);
	  }
  } 
}

template <typename Dtype>
void FreezeDropPathLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void FreezeDropPathLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->phase_ == TRAIN) {
	branch = 0;
	if (stochastic) {  
      uint rnd = std::rand() % uint_thresholds_[bottom.size()];
      for (int i = 0; i < bottom.size(); ++i) {
		  if (rnd >= uint_thresholds_[i] && rnd < uint_thresholds_[i+1]) branch = i;
      }
    } else {
		iteration = (iteration + 1) % this->layer_param_.freeze_drop_path_param().num_iter_per_cycle();
        for (int i = 0; i < bottom.size(); ++i) {
		  if (iteration >= thresholds_[i] && iteration < thresholds_[i+1]) branch = i;
		}
		if (branch > lastBranch || branch < lastBranch) 
		  LOG(INFO) << " Changing from branch " << lastBranch << " to " << branch;
	}
  } else {
	  branch = bottom.size() - 1;
  }
  
//	Zero the top layer before adding the bottom data
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  for (int i = 0; i <= branch; ++i) {
      caffe_axpy(top[0]->count(), Dtype(1.0), bottom[i]->cpu_data(), top[0]->mutable_cpu_data());
  }
  lastBranch = branch;
}

template <typename Dtype>
void FreezeDropPathLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      if (i == branch) {
        caffe_copy(bottom[i]->count(), top[0]->cpu_diff(), bottom[i]->mutable_cpu_diff());
      } else {
        caffe_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_cpu_diff());
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FreezeDropPathLayer);
#endif

INSTANTIATE_CLASS(FreezeDropPathLayer);
REGISTER_LAYER_CLASS(FreezeDropPath);

}  // namespace caffe


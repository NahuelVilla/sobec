///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SOBEC_STATELPF_FACTORY_HPP_
#define SOBEC_STATELPF_FACTORY_HPP_

#include <crocoddyl/core/numdiff/state.hpp>
#include <crocoddyl/core/state-base.hpp>
#include <crocoddyl/core/utils/exception.hpp>

#include "pinocchio_model.hpp"
#include "sobec/lowpassfilter/statelpf.hpp"

namespace sobec {
namespace unittest {

struct StateLPFModelTypes {
  enum Type {
    StateLPF_TalosArm,
    StateLPF_HyQ,
    StateLPF_Talos,
    StateLPF_RandomHumanoid,
    NbStateLPFModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbStateLPFModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, StateLPFModelTypes::Type type);

class StateLPFModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit StateLPFModelFactory();
  ~StateLPFModelFactory();

  boost::shared_ptr<sobec::StateLPF> create(
      StateLPFModelTypes::Type state_type) const;
};

}  // namespace unittest
}  // namespace sobec

#endif  // SOBEC_STATELPF_FACTORY_HPP_

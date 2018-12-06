// @HEADER
// ************************************************************************
//
//                           Compadre Package
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
// IN NO EVENT SHALL SANDIA CORPORATION OR THE CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Paul Kuberry  (pakuber@sandia.gov)
//                    Peter Bosler  (pabosle@sandia.gov), or
//                    Nat Trask     (natrask@sandia.gov)
//
// ************************************************************************
// @HEADER
// this file picks up at the beginning of the computeTargetFunctionals function
#ifndef _USER_STANDARD_TARGET_FUNCTIONALS_HPP_
#define _USER_STANDARD_TARGET_FUNCTIONALS_HPP_

bool some_conditions_for_a_user_defined_operation = false;
bool some_conditions_for_another_user_defined_operation = false;

// hint: look in Compadre_GMLS_Target.hpp for examples

if (some_conditions_for_a_user_defined_operation) {
    // these operations are being called at the Team level,
    // so we call single to only perform the operation on one thread
    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
        // user definition for a target functional goes here


    });
} else if (some_conditions_for_another_user_defined_operation) {
    // these operations are being called at the Team level,
    // so we call single to only perform the operation on one thread
    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
        // user definition for a different target functional goes here
        

    });
} else {
    // if the operation was not caught by any user defined TargetFunctional,
    // then it is returned to the toolkit to try to handle the operation
    operation_handled = false;
}

#endif

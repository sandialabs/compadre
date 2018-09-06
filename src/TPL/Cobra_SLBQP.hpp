/** \file   Cobra_SLBQP.hpp
    \brief  Header file for the optimization algorithms for
            singly linearly constrained quadratic programs
            with simple bounds, geared toward optimization-
            based remap.
            Contains the algorithm function as well as a
            template for the algebra interface used by the
            algorithm.
            COBRA stands for:
            Constrained Optimization-Based Remap Algorithms.
    \author Created by D. Ridzal (Sandia National Labs).
*/


#ifndef COBRA_SLBQP_HPP
#define COBRA_SLBQP_HPP

/*!
 *  \addtogroup Cobra
 *  @{
 */

//! Generic optimization interfaces and implementations.
namespace Cobra {



/** \class Cobra::AlgebraInterface
    \brief Defines the template for the algebra interface
           for the SLBQP algorithm.  Only three operations
           need be defined: median, axpy and dot product.
*/
template<class Scalar, class Vector>
class AlgebraInterface {

public:

  /** \brief Compute entry-wise median of three vectors:
             med, low, upp.

      \param med  [in/out]  - Result vector and first input vector.
      \param low  [in]      - Second input vector.
      \param upp  [in]      - Third input vector.
  */
  virtual void median(Vector & med,
                      const Vector & low,
                      const Vector & upp) = 0;

  /** \brief Returns the vector dot product (l2 inner product).
      
      \param  x [in]  - First input vector.
      \param  y [in]  - Second input vector.
  */
  virtual Scalar vdot(const Vector & x,
                      const Vector & y) = 0;

  /** \brief Computes z = a*x+y.

      \param  z [out] - Result vector.
      \param  a [in]  - Scalar multiplier.
      \param  x [in]  - First input vector.
      \param  y [in]  - Second input vector.
  */
  virtual void zaxpy(Vector & z,
                     Scalar a,
                     const Vector & x,
                     const Vector & y) = 0;

};



/** \struct Cobra::AlgoStatus
    \brief Defines the return struct to get the algorithm status.
*/
template<class Scalar>
struct AlgoStatus {

  /** \brief  Contains convergence status (true/false).
  */
  bool   converged;

  /** \brief  Contains total number of iterations.
  */
  int    iterations;

  /** \brief  Contains number of bracketing iterations.
  */
  int    bracketing_iterations;

  /** \brief  Contains number of secant iterations.
  */
  int    secant_iterations;

  /** \brief  Contains Lagrange multiplier for equality constraint
              (mass conservation constraint).
  */
  Scalar lambda;

  /** \brief  Contains equality constraint residual
              (mass conservation constraint).
  */
  Scalar residual;

};



/** \fn    bool Cobra::slbqp_local(AlgoStatus<Scalar> & astatus,
                                   Vector & rho,
                                   const Vector & rho_t,
                                   const Vector & lower,
                                   const Vector & upper,
                                   const Vector & rowsum,
                                   Scalar mass,
                                   AlgebraInterface<Scalar, Vector> * aif) 
    \brief Solves singly linearly constrained qudratic programs
           with simple bounds, for optimization-based remap;
           ASSUMES A GOOD LAGRANGE MULTIPLIER GUESS FOR
           THE LINEAR CONSTRAINT.
           In other words, global convergence, i.e.,
           convergence from remote starting points, is not
           ensured.  Use this algorithm if speed is of the
           essence and if you are confident that the objective
           function target is fairly close to the minimizer.

    \param  astatus [out]  - Returns algorithm status.
    \param  rho [out]      - Returns result density vector.
    \param  rho_t [in]     - Target density vector.
    \param  lower [in]     - Vector of lower bounds (local mins).
    \param  upper [in]     - Vector of upper bounds (local maxes).
    \param  rowsum [in]    - Vector of mass-matrix rowsums.
    \param  mass [in]      - Scalar mass total.
    \param  aif [in]       - Pointer to user-defined algebra interface.

    The return value is the convergence flag,
    true = converged, false = not converged.
*/
template<class Scalar, class Vector>
bool slbqp_local(AlgoStatus<Scalar> & astatus,
                 Vector & rho,
                 const Vector & rho_t,
                 const Vector & lower,
                 const Vector & upper,
                 const Vector & rowsum,
                 Scalar mass,
                 AlgebraInterface<Scalar, Vector> * aif) {

  // Set some algorithm-specific constants and temporaries.
  Scalar lambda   = 0;
  Scalar hfd      = 1e-8;
  Scalar eta      = 1e-12;
  Scalar eqtol    = 1e-14;
  int maxiter     = 30;
  bool ignoremass = false;
  int nclip       = 0;
  Scalar rp       = 0;
  Scalar rc       = 0;
  Scalar alpha    = 0;
  Scalar lambda_p = 0;
  Scalar lambda_c = 0;

  /*********** Secant algorithm. ***********/

  // Solve QP with fixed lambda:
  aif->zaxpy(rho, lambda, rowsum, rho_t);  // rho = lambda*rowsum + rho_t
  aif->median(rho, lower, upper);          // rho = median(rho, lower, upper)
  // Compute residual:
  rp = aif->vdot(rowsum,rho) - mass;
  // Increment counter:
  nclip += 1;

  if ( ((rp < eta) && (-rp < eta)) || ignoremass ) {  // check |rp| < eta --> converged; OR only enforce bounds
    astatus.iterations = nclip;
    astatus.lambda     = lambda;
    astatus.residual   = rp;
    astatus.converged  = true;
    return true;
  }

  // Solve QP with fixed lambda:
  aif->zaxpy(rho, lambda+hfd, rowsum, rho_t);  // rho = (lambda+hfd)*rowsum + rho_t
  aif->median(rho, lower, upper);              // rho = median(rho, lower, upper)
  // Compute residual:
  rc = aif->vdot(rowsum,rho) - mass;
  // Increment counter:
  nclip += 1;

  // First compute finite difference slope.
  if ( (((rc - rp) < eqtol) && ((rp - rc) < eqtol)) &&  // check rc == rp   --> stagnation
       ((rc > eta) || (-rc > eta)) ) {                  // check |rc| > eta --> not converged
    astatus.iterations = nclip;
    astatus.lambda     = lambda;
    astatus.residual   = rp;
    astatus.converged  = false;
    return false;
  }
  alpha = hfd / (rc - rp);

  // Set initial step, compute second step based on the
  // finite difference approximation
  lambda_p = lambda;
  lambda_c = lambda_p - alpha*rp;

  while ( ((rc > eta) || (-rc > eta)) && (nclip < maxiter) ) {  // while not converged
    // Solve QP with fixed lambda:
    aif->zaxpy(rho, lambda_c, rowsum, rho_t);  // rho = (lambda+hfd)*rowsum + rho_t
    aif->median(rho, lower, upper);            // rho = median(rho, lower, upper)
    // Compute residual:
    rc = aif->vdot(rowsum,rho) - mass;
    // Increment counter:
    nclip += 1;

    // Compute slope.
    if ( (((rc - rp) < eqtol) && ((rp - rc) < eqtol)) &&  // check rc == rp   --> stagnation
         ((rc > eta) || (-rc > eta)) ) {                  // check |rc| > eta --> not converged
      astatus.iterations = nclip;
      astatus.lambda     = lambda_c;
      astatus.residual   = rc;
      astatus.converged  = false;
      return false;
    }
    alpha = (lambda_p - lambda_c) / (rp - rc);
    rp = rc;

    // Take step.
    lambda_p   = lambda_c;
    lambda_c   = lambda_c - alpha*rc;
  }

  astatus.iterations            = nclip;
  astatus.secant_iterations     = astatus.iterations;
  astatus.bracketing_iterations = 0;
  astatus.lambda                = lambda_c;
  astatus.residual              = rc;
  if ((rc > eta) || (-rc > eta)) {  // check |rc| > eta --> not converged
    astatus.converged  = false;
  }
  else {
    astatus.converged  = true;
  }

  return astatus.converged;

};



/** \fn    bool Cobra::slbqp(AlgoStatus<Scalar> & astatus,
                             Vector & rho,
                             const Vector & rho_t,
                             const Vector & lower,
                             const Vector & upper,
                             const Vector & rowsum,
                             Scalar mass,
                             AlgebraInterface<Scalar, Vector> * aif) 
    \brief Solves singly linearly constrained qudratic programs
           with simple bounds, for optimization-based remap.
           We use the bracketing/secant procedure proposed in
           Dai, Fletcher: New algorithms for singly linearly
           constrained quadratic programs subject to lower and
           upper bounds, Math. Program. (A) 106, 403-421 (2006).

    \param  astatus [out]  - Returns algorithm status.
    \param  rho [out]      - Returns result density vector.
    \param  rho_t [in]     - Target density vector.
    \param  lower [in]     - Vector of lower bounds (local mins).
    \param  upper [in]     - Vector of upper bounds (local maxes).
    \param  rowsum [in]    - Vector of mass-matrix rowsums.
    \param  mass [in]      - Scalar mass total.
    \param  aif [in]       - Pointer to user-defined algebra interface.

    The return value is the convergence flag,
    true = converged, false = not converged.
*/
template<class Scalar, class Vector>
bool slbqp(AlgoStatus<Scalar> & astatus,
           Vector & rho,
           const Vector & rho_t,
           const Vector & lower,
           const Vector & upper,
           const Vector & rowsum,
           Scalar mass,
           AlgebraInterface<Scalar, Vector> * aif) {

  // Set some algorithm-specific constants and temporaries.
  Scalar eta      = 1e-12;
  int maxiter     = 30;
  bool ignoremass = false;
  int nclip       = 0;
  Scalar l        = 0;
  Scalar llow     = 0;
  Scalar lupp     = 0;
  Scalar lnew     = 0;
  Scalar dl       = 2;
  Scalar r        = 0;
  Scalar rlow     = 0;
  Scalar rupp     = 0;
  Scalar s        = 0;


  /********** Start bracketing phase of SLBQP **********/

  // Solve QP with fixed Lagrange multiplier:
  aif->zaxpy(rho, l, rowsum, rho_t);  // rho = l*rowsum + rho_t
  aif->median(rho, lower, upper);     // rho = median(rho, lower, upper)
  // Compute residual:
  r = aif->vdot(rowsum,rho) - mass;
  // Increment counter:
  nclip += 1;

  if (r < 0) {

    llow = l;  rlow = r;  l = l + dl;

    // Solve QP with fixed Lagrange multiplier:
    aif->zaxpy(rho, l, rowsum, rho_t);  // rho = l*rowsum + rho_t
    aif->median(rho, lower, upper);     // rho = median(rho, lower, upper)
    // Compute residual:
    r = aif->vdot(rowsum,rho) - mass;
    // Increment counter:
    nclip += 1;

    while ((r < 0) && (nclip < maxiter)) {
      llow = l;
      s = rlow/r - 1.0;  if (s < 0.1) {s = 0.1;}
      dl = dl + dl/s;  l = l + dl;
      // Solve QP with fixed Lagrange multiplier:
      aif->zaxpy(rho, l, rowsum, rho_t);  // rho = l*rowsum + rho_t
      aif->median(rho, lower, upper);     // rho = median(rho, lower, upper)
      // Compute residual:
      r = aif->vdot(rowsum,rho) - mass;
      // Increment counter:
      nclip += 1;
    }

    lupp = l;  rupp = r;
  }

  else {

    lupp = l;  rupp = r;  l = l - dl;

    // Solve QP with fixed Lagrange multiplier:
    aif->zaxpy(rho, l, rowsum, rho_t);  // rho = l*rowsum + rho_t
    aif->median(rho, lower, upper);     // rho = median(rho, lower, upper)
    // Compute residual:
    r = aif->vdot(rowsum,rho) - mass;
    // Increment counter:
    nclip += 1;

    while ((r > 0) && (nclip < maxiter)) {
      lupp = l;
      s = rupp/r - 1.0;  if (s < 0.1) {s = 0.1;}
      dl = dl + dl/s;  l = l - dl;
      // Solve QP with fixed Lagrange multiplier:
      aif->zaxpy(rho, l, rowsum, rho_t);  // rho = l*rowsum + rho_t
      aif->median(rho, lower, upper);     // rho = median(rho, lower, upper)
      // Compute residual:
      r = aif->vdot(rowsum,rho) - mass;
      // Increment counter:
      nclip += 1;
    }

    llow = l;  rlow = r;
  }

  astatus.bracketing_iterations = nclip;

  /********** Stop bracketing phase of SLBQP **********/


  /********** Start secant phase of SLBQP **********/

  s = 1.0 - rlow/rupp;  dl = dl/s;  l = lupp - dl;
 
  // Solve QP with fixed Lagrange multiplier:
  aif->zaxpy(rho, l, rowsum, rho_t);  // rho = l*rowsum + rho_t
  aif->median(rho, lower, upper);     // rho = median(rho, lower, upper)
  // Compute residual:
  r = aif->vdot(rowsum,rho) - mass;
  // Increment counter:
  nclip += 1;

  while ( ((r > eta) || (-r > eta)) && (nclip < maxiter) ) {  // while not converged  

    if (r > 0) {

      if (s <= 2.0) {
        lupp = l;  rupp = r;  s = 1.0 - rlow/rupp;
        dl = (lupp - llow)/s;  l = lupp - dl;
      }
      else {
        s = rupp/r - 1.0;   if (s < 0.1) {s = 0.1;}
        dl = (lupp - l)/s;
        lnew = 0.75*llow + 0.25*l;  if (lnew < l-dl) {lnew = l-dl;}
        lupp = l;  rupp = r;  l = lnew;
        s = (lupp - llow)/(lupp - l);
      }

    }

    else {

      if (s >= 2.0) {
        llow = l;  rlow = r;  s = 1.0 - rlow/rupp;
        dl = (lupp - llow)/s;  l = lupp - dl;
      }
      else {
	s = rlow/r - 1.0;   if (s < 0.1) {s = 0.1;}
       	dl = (l - llow)/s;
       	lnew = 0.75*lupp + 0.25*l;  if (lnew < l+dl) {lnew = l+dl;}
       	llow = l;  rlow = r; l = lnew;
       	s = (lupp - llow)/(lupp - l);
      }

    }

    // Solve QP with fixed Lagrange multiplier:
    aif->zaxpy(rho, l, rowsum, rho_t);  // rho = l*rowsum + rho_t
    aif->median(rho, lower, upper);     // rho = median(rho, lower, upper)
    // Compute residual:
    r = aif->vdot(rowsum,rho) - mass;
    // Increment counter:
    nclip += 1;

  }

  /********** Stop secant phase of SLBQP **********/

  astatus.iterations = nclip;
  astatus.secant_iterations = astatus.iterations - astatus.bracketing_iterations;
  astatus.lambda     = l;
  astatus.residual   = r;
  if ((r > eta) || (-r > eta)) {  // check |r| > eta --> not converged
    astatus.converged  = false;
  }
  else {
    astatus.converged  = true;
  }

  return astatus.converged;

}

class StdVectorInterface : public Cobra::AlgebraInterface<double, std::vector<double> > {

public:

  StdVectorInterface(){ };

  ~StdVectorInterface(){ };

  // entry-wise median
  void median(std::vector<double> & med,
              const std::vector<double> & low,
              const std::vector<double> & upp) {;

    for (int i=0; i<med.size(); i++) {
      // med >= low and med <= upp OR med <= low and med >= upp
      if ( (med[i] - low[i]) * (upp[i] - med[i]) >= 0 )
                       ;  // med[i] is already the median, i.e., med[i] = med[i]

      // low >= med and low <= upp OR low <= med and low >= upp
      else if ( (low[i] - med[i]) * (upp[i] - low[i]) >= 0 )
        med[i] = low[i];  // low[i] is the median

      else
        med[i] = upp[i];  // upp[i] is the median
    }

  }

  // vector dot product (l2 inner product)
  double vdot(const std::vector<double> & x,
              const std::vector<double> & y) {;

    double sum = 0;

    for (int i=0; i<x.size(); i++) {
      sum += x[i]*y[i];
    }

    return sum;

  }

  // z = a*x+y
  void zaxpy(std::vector<double> & z,
             double a,
             const std::vector<double> & x,
             const std::vector<double> & y) {

    for (int i=0; i<z.size(); i++) {
      z[i] = a*x[i]+y[i];
    }

  }

};

}  // namespace Cobra

/*! @} End of Doxygen Groups*/

#endif

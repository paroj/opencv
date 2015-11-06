/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <stdio.h>

/*
   This is translation to C++ of the Matlab's LMSolve package by Miroslav Balda.
   Here is the original copyright:
   ============================================================================

   Copyright (c) 2007, Miroslav Balda
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in
         the documentation and/or other materials provided with the distribution

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

namespace cv
{

class LMSolverImpl : public LMSolver
{
public:
    LMSolverImpl() : maxIters(100) { init(); }
    LMSolverImpl(const Ptr<LMSolver::Callback>& _cb, int _maxIters) : cb(_cb), maxIters(_maxIters) { init(); }

    void init()
    {
        epsx = epsf = FLT_EPSILON;
        printInterval = 0;
    }

    /*
     * implementation of
     * Fletcher, Roger. MODIFIED MARQUARDT SUBROUTINE FOR NON-LINEAR LEAST SQUARES.
     * No. AERE-R--6799. Atomic Energy Research Establishment, Harwell (England), 1971.
     */
    int run(InputOutputArray _param0) const
    {
        Mat param0 = _param0.getMat(), x, xd, r, rd, J, A, Ap, v, temp_d, d;
        int ptype = param0.type();

        CV_Assert( (param0.cols == 1 || param0.rows == 1) && (ptype == CV_32F || ptype == CV_64F));
        CV_Assert( cb );

        int lx = param0.rows + param0.cols - 1;
        param0.convertTo(x, CV_64F);

        if( x.cols != 1 )
            transpose(x, x);

        if( !cb->compute(x, r, J) )
            return -1;
        double S = norm(r, NORM_L2SQR);
        int nfJ = 2;

        mulTransposed(J, A, true);
        gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);

        Mat D = A.diag().clone();

        const double Rlo = 0.25, Rhi = 0.75;
        double lambda = 1, lc = 0.75;
        int i, iter = 0;

        if( printInterval != 0 )
        {
            printf("************************************************************************************\n");
            printf("\titr\tnfJ\t\tSUM(r^2)\t\tx\t\tdx\t\tl\t\tlc\n");
            printf("************************************************************************************\n");
        }

        for( ;; )
        {
            CV_Assert( A.type() == CV_64F && A.rows == lx );
            A.copyTo(Ap);
            for( i = 0; i < lx; i++ )
                Ap.at<double>(i, i) += lambda*D.at<double>(i);
            solve(Ap, v, d, DECOMP_EIG);
            subtract(x, d, xd);
            if( !cb->compute(xd, rd, noArray()) )
                return -1;
            nfJ++;
            double Sd = norm(rd, NORM_L2SQR);
            gemm(A, d, -1, v, 2, temp_d);
            double dS = d.dot(temp_d);
            double R = (S - Sd)/(fabs(dS) > DBL_EPSILON ? dS : 1);

            if( R > Rhi )
            {
                lambda *= 0.5;
                if( lambda < lc )
                    lambda = 0;
            }
            else if( R < Rlo )
            {
                // find new nu if R too low
                double t = d.dot(v);
                double nu = (Sd - S)/(fabs(t) > DBL_EPSILON ? t : 1) + 2;
                nu = std::min(std::max(nu, 2.), 10.);
                if( lambda == 0 )
                {
                    invert(A, Ap, DECOMP_EIG);
                    double maxval = DBL_EPSILON;
                    for( i = 0; i < lx; i++ )
                        maxval = std::max(maxval, std::abs(Ap.at<double>(i,i)));
                    lambda = lc = 1./maxval;
                    nu *= 0.5;
                }
                lambda *= nu;
            }

            if( Sd < S )
            {
                nfJ++;
                S = Sd;
                std::swap(x, xd);
                if( !cb->compute(x, r, J) )
                    return -1;
                mulTransposed(J, A, true);
                gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);
            }

            iter++;
            bool proceed = iter < maxIters && norm(d, NORM_INF) >= epsx && norm(r, NORM_INF) >= epsf;

            if( printInterval != 0 && (iter % printInterval == 0 || iter == 1 || !proceed) )
            {
                printf("%c%10d %10d %15.4e %16.4e %17.4e %16.4e %17.4e\n",
                       (proceed ? ' ' : '*'), iter, nfJ, S, x.at<double>(0), d.at<double>(0), lambda, lc);
            }

            if(!proceed)
                break;
        }

        if( param0.size != x.size )
            transpose(x, x);

        x.convertTo(param0, ptype);
        if( iter == maxIters )
            iter = -iter;

        return iter;
    }

    void setCallback(const Ptr<LMSolver::Callback>& _cb) { cb = _cb; }

    Ptr<LMSolver::Callback> cb;

    double epsx;
    double epsf;
    int maxIters;
    int printInterval;
};


Ptr<LMSolver> createLMSolver(const Ptr<LMSolver::Callback>& cb, int maxIters)
{
    return makePtr<LMSolverImpl>(cb, maxIters);
}

SparseLevMarq::SparseLevMarq(const TermCriteria& criteria0 )
{
    errNorm = prevErrNorm = DBL_MAX;
    lambdaLg10 = -3;
    criteria = criteria0;
    state = STARTED;
    iters = 0;
}

bool SparseLevMarq::updateAlt( Mat& _U, Mat& _V, Mat& _W, Mat& _JtErr, double*& _errNorm )
{
    if( state == DONE )
    {
        return false;
    }

    if( state == STARTED )
    {
        param.copyTo( prevParam );
        errNorm = 0;
        _errNorm = &errNorm;
        state = CALC_J;
        return true;
    }

    if( state == CALC_J )
    {
        W = _W;
        JtErr = _JtErr;
        U = _U;
        V = _V;
        param.copyTo( prevParam );
        step();
        prevErrNorm = errNorm;
        errNorm = 0;
        _errNorm = &errNorm;
        state = CHECK_ERR;
        return true;
    }

    assert( state == CHECK_ERR );
    if( errNorm > prevErrNorm )
    {
        if( ++lambdaLg10 <= 16 )
        {
            step();
            errNorm = 0;
            _errNorm = &errNorm;
            state = CHECK_ERR;
            return true;
        }
    }

    lambdaLg10 = MAX(lambdaLg10-1, -16);
    if( ++iters >= criteria.maxCount ||
        (norm(param, prevParam, NORM_L2 | NORM_RELATIVE)) < criteria.epsilon )
    {
        state = DONE;
        return false;
    }

    prevErrNorm = errNorm;
    state = CALC_J;
    return true;
}

/**
 * implements HZ: A6.3
 */
void SparseLevMarq::step()
{
    const double LOG10 = log(10.);
    double lambda = exp(lambdaLg10*LOG10);
    int nparams = param.rows;

    for( int i = 0; i < nparams; i++ )
        if( !mask.ptr()[i] )
        {
            U.row(i) = 0;
            U.col(i) = 0;
            W.row(i) = 0;
            JtErr.at<double>(i) = 0;
        }

    const int nparamsA = U.cols;
    const int nparamsB = V.cols;
    const int nblocksB = (nparams - nparamsA) / nparamsB;

    CV_Assert((nparams - nparamsA) % nparamsB == 0);

    // HZ A6.3 (ii)
    U.copyTo(Ustar);
    V.copyTo(Vstar);

#if 1
    Ustar.diag() *= 1. + lambda;
    for(int i = 0; i < nblocksB; i++) {
        Vstar.rowRange(i * nparamsB, (i + 1) * nparamsB).diag() *= 1. + lambda;
    }
#else
    Ustar.diag() += lambda;
    for(int i = 0; i < nblocksB; i++) {
        Vstar.colRange(i * nparamsB, (i + 1) * nparamsB).diag() += lambda;
    }
#endif

    Y.create(nparamsA*nblocksB, nparamsB, CV_64F);

    for(int i = 0; i < nblocksB; i++) {
        // Y = W . V*^-1
        Y.rowRange(i*nparamsA, (i + 1)*nparamsA) =
                W.colRange(i * nparamsB, (i + 1) * nparamsB)*Vstar.rowRange(i * nparamsB, (i + 1) * nparamsB).inv();
    }

    // HZ A6.3 (iii)
    Mat S = Ustar;
    Mat e = JtErr.rowRange(0, nparamsA).clone();

    const Mat errB = JtErr.rowRange(nparamsA, JtErr.rows);

    for(int i = 0; i < nblocksB; i++) {
        S -= Y.rowRange(i * nparamsA, (i + 1) * nparamsA) * W.colRange(i * nparamsB, (i + 1) * nparamsB).t();
        e -= Y.rowRange(i * nparamsA, (i + 1) * nparamsA) * errB.rowRange(i * nparamsB, (i + 1) * nparamsB);
    }

    const Mat paramA = param.rowRange(0, nparamsA);

    solve(S, e, paramA, DECOMP_SVD);

    // HZ A6.3 (iv)
    for(int i = 0; i < nblocksB; i++) {
        param.rowRange(nparamsA + i*nparamsB, nparamsA + (i + 1)*nparamsB) =
                Vstar.rowRange(i * nparamsB, (i + 1) * nparamsB).inv()
                *(errB.rowRange(i * nparamsB, (i + 1) * nparamsB) -
                        W.colRange(i * nparamsB, (i + 1) * nparamsB).t()*paramA);
    }

    for( int i = 0; i < nparams; i++ )
        param.at<double>(i) = prevParam.at<double>(i) - (mask.ptr()[i] ? param.at<double>(i) : 0);
}

}

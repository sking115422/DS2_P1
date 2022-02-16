

package scalation
package modeling

import scala.collection.mutable.Set
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._



//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymbolicRegression` object supports symbolic regression that allows
 *  variables/columns to be raised to various powers, e.g., x^2, x^3, x^.5.
 *  IMPORTANT:  must not include intercept (column of ones) in initial data matrix),
 *  i.e., DO NOT include a column of ones in x (will cause singularity if intercept is true).
 */
object SymbolicRegression:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Regression` object from a data matrix and a response vector.
     *  Partial support for "Symbolic Regression" as matrix x can be raised
     *  to several powers (e.g., x^1 and x^2).  Note, x^1 is automatically included.
     *  @param x          the initial data/input m-by-n matrix (before expansion)
     *                        must not include an intercept column of all ones
     *  @param y          the response/output m-vector
     *  @param fname      the feature/variable names (use null for default)
     *  @param powers     the set of powers to raise matrix x to
     *  @param intercept  whether to include the intercept term (column of ones) _1 (defaults to true)
     *  @param cross      whether to include 2-way cross/interaction terms x_i x_j (defaults to true)
     *  @param cross3     whether to include 3-way cross/interaction terms x_i x_j x_k (defaults to false)
     *  @param hparam     the hyper-parameters (use Regression.hp for default)
     */
    def apply (x: MatrixD, y: VectorD, fname: Array [String],
               powers: Set [Double], intercept: Boolean = true,
               cross: Boolean = true, cross3: Boolean = false,
               hparam: HyperParameter = Regression.hp): Regression =
        var xx     = x                                                    // start with multiple regression for x
        var fname_ = fname

        for p <- powers if p != 1 do
            xx       = xx ++^ x~^p                                        // add power terms x^p
            fname_ ++= fname.map ((n) => s"$n^${p.toInt}")
        end for

        if intercept then
            xx     = MatrixD.one (x.dim) ++^ xx                           // add intercept term (column of ones) _1
            fname_ = Array ("intercept") ++ fname_
        end if

        if cross then
           xx       = xx ++^ x.crossAll                                   // add 2-way cross terms x_i x_j
           fname_ ++= crossNames (fname)
        end if

        if cross3 then
           xx       = xx ++^ x.crossAll3                                  // add 3-way cross terms x_i x_j x_k
           fname_ ++= crossNames3 (fname)
        end if

        val mod = new Regression (xx, y, fname_, hparam)
        mod.modelName = "SymbolicRegression" + (if cross then "X" else "") + (if cross3 then "X" else "")
        mod
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `SymbolicRegression` object from a data matrix and a response vector.
     *  This factory function provides data rescaling.
     *  @param x          the data/input m-by-n matrix
     *                        (augment with a first column of ones to include intercept in model)
     *  @param y          the response/output m-vector
     *  @param powers     the set of powers to raise matrix x to
     *  @param intercept  whether to include the intercept term (column of ones) _1 (defaults to true)
     *  @param cross      whether to include 2-way cross/interaction terms x_i x_j (defaults to true)
     *  @param cross3     whether to include 3-way cross/interaction terms x_i x_j x_k (defaults to false)
     *  @param fname      the feature/variable names (use null for default)
     *  @param hparam     the hyper-parameters (use Regression.hp for default)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 powers: Set [Double], intercept: Boolean = true,
                 cross: Boolean = true, cross3: Boolean = false,
                 hparam: HyperParameter = Regression.hp): Regression =
        val xn = normalize (x, (x.mean, x.stdev))
        apply (xn, y, fname, powers, intercept, cross, cross3, hparam)
    end rescale

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create all cross names for the 2-way interaction/cross terms: e.g., "name1_name2".
     *  @param nm  the array of names to be crossed
     */
    def crossNames (nm: Array [String]): Array [String] =
        (for i <- nm.indices; j <- 0 until i yield s"${nm(i)}_${nm(j)}").toArray
    end crossNames

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create all cross names for the 3-way interaction/cross terms: e.g., "name1_name2_name3".
     *  @param nm  the array of names to be crossed
     */
    def crossNames3 (nm: Array [String]): Array [String] =
        (for i <- nm.indices; j <- 0 until i; k <- 0 until j yield s"${nm(i)}_${nm(j)}_${nm(k)}").toArray
    end crossNames3

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Regression` object that uses multiple regression to fit a quadratic
     *  surface to the data.  For example in 2D, the quadratic regression equation is
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_0, x_0^2, x_1, x_1^2] + e
     *  @param x          the initial data/input m-by-n matrix (before quadratic term expansion)
     *                        must not include an intercept column of all ones
     *  @param y          the response/output m-vector
     *  @param fname      the feature/variable names (use null for default)
     *  @param intercept  whether to include the intercept term (column of ones) _1 (defaults to true)
     *  @param cross      whether to include 2-way cross/interaction terms x_i x_j (defaults to false)
     *  @param hparam     the hyper-parameters ((use Regression.hp for default)
     */
    def quadratic (x: MatrixD, y: VectorD, fname: Array [String],
                   intercept: Boolean = true, cross: Boolean = false,
                   hparam: HyperParameter = Regression.hp): Regression =
        val mod = apply (x, y, fname, Set (2), intercept, cross, false, hparam)
        mod.modelName = "SymbolicRegression.quadratic" + (if cross then "X" else "")
        mod
    end quadratic

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Regression` object that uses multiple regression to fit a cubic
     *  surface to the data.  For example in 2D, the cubic regression equation is
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_0, x_0^2, x_0^3,
     *                                                   x_1, x_1^2, x_1^3,
     *                                                   x_0*x_1, x_0^2*x_1, x_0*x_1^2] + e
     *  @param x          the initial data/input m-by-n matrix (before quadratic term expansion)
     *                        must not include an intercept column of all ones
     *  @param y          the response/output m-vector
     *  @param fname      the feature/variable names (use null for default)
     *  @param intercept  whether to include the intercept term (column of ones) _1 (defaults to true)
     *  @param cross      whether to include 2-way cross/interaction terms x_i x_j (defaults to false)
     *  @param cross3     whether to include 3-way cross/interaction terms x_i x_j x_k (defaults to false)
     *  @param hparam     the hyper-parameters ((use Regression.hp for default)
     */
    def cubic (x: MatrixD, y: VectorD, fname: Array [String],
               intercept: Boolean = true, cross: Boolean = false, cross3: Boolean = false,
               hparam: HyperParameter = Regression.hp): Regression =
        val mod = apply (x, y, fname, Set (2, 3), intercept, cross, cross3, hparam)
        mod.modelName = "SymbolicRegression.cubic" + (if cross then "X" else "") + (if cross3 then "X" else "")
        mod
    end cubic

end SymbolicRegression

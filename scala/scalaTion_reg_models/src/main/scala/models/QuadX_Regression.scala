

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Dec 23 13:54:30 EST 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Quadractic Regression, including Quadratic and Cubic Regression
 */

package scalation
package modeling

import scala.collection.mutable.Set
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `symbolicRegressionTest` main function tests the `SymbolicRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Quadractic Regression", with powers specified in "Set (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.symbolicRegressionTest
 */
@main def QuadX_Regression_AutoMPG (): Unit =
    
    import AutoMPG_Data._

    // banner ("Variable Names in AutoMPG Dataset")
    // println (s"xr_fname = ${stringOf (xr_fname)}")                     // raw dataset
    // println (s"x_fname  = ${stringOf (x_fname)}")                      // origin column removed
    // println (s"ox_fname = ${stringOf (ox_fname)}")                     // intercept (1's) added

    // println (s"x = $x")
    // println (s"y = $y")
    
    banner ("Auto MPG Quadractic Regression")
    val mod = SymbolicRegression.quadratic(x, y, x_fname)    // add cross-terms and given powers
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Quadractic Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end QuadX_Regression_AutoMPG



//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymbolicRegressionTest` main function tests the `SymbolicRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Quadractic Regression", with powers specified in "Set (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymbolicRegressionTest
 */
@main def QuadX_Regression_ForestFires (): Unit =
    
    import ForestFires_Data._
    
    banner ("Forest Fires Quadractic Regression")                     
    val mod = SymbolicRegression.quadratic(x, y, x_fname)         // add cross-terms and given powers
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Quadractic Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end QuadX_Regression_ForestFires



//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymbolicRegressionTest` main function tests the `SymbolicRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Quadractic Regression", with powers specified in "Set (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymbolicRegressionTest
 */
@main def QuadX_Regression_AirQuality (): Unit =
    
    import AirQuality_Data._
    
    banner ("Air Quality Quadractic Regression")
    val mod = SymbolicRegression.quadratic(x, y, x_fname)    // add cross-terms and given powers
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Quadractic Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end QuadX_Regression_AirQuality



//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymbolicRegressionTest` main function tests the `SymbolicRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Quadractic Regression", with powers specified in "Set (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymbolicRegressionTest
 */
@main def QuadX_Regression_CCPP (): Unit =
    
    import CCPP_Data._
    
    banner ("CCPP Quadractic Regression")
    val mod = SymbolicRegression.quadratic(x, y, x_fname)    // add cross-terms and given powers
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Quadractic Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end QuadX_Regression_CCPP



//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymbolicRegressionTest` main function tests the `SymbolicRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Quadractic Regression", with powers specified in "Set (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymbolicRegressionTest
 */
@main def QuadX_Regression_WineQuality (): Unit =
    
    import WineQuality_Data._
    
    banner ("Wine Quality Quadractic Regression")
    val mod = SymbolicRegression.quadratic(x, y, x_fname)    // add cross-terms and given powers
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Quadractic Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end QuadX_Regression_WineQuality



//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymbolicRegressionTest` main function tests the `SymbolicRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Quadractic Regression", with powers specified in "Set (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymbolicRegressionTest
 */
@main def QuadX_Regression_BikeSharing (): Unit =
    
    import BikeSharing_Data._
    
    banner ("Bike Sharing Quadractic Regression")
    val mod = SymbolicRegression.quadratic(x, y, x_fname)         // add cross-terms and given powers
                                                                          //false, true, true)) // no intercept, 2&3-way cross terms
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Quadractic Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end QuadX_Regression_BikeSharing
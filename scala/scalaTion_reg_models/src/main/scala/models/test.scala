

package scalation
package modeling

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._


@main def test (): Unit = {

    val auto_mat = MatrixD.load("forestfires_cleaned.csv")
    println(auto_mat)

    val x = auto_mat(?, 1 to 12)
    val y = auto_mat(?, 13)

    println (x)
    println (y)
}



// import scalation.database.relation.Relation

// banner ("auto_mpg data")
// val auto_tab = Relation ("auto_mpg_fixed_cleaned.csv", "auto_mpg", null, -1)
// auto_tab.show ()

// val auto_mat = MatrixD.load("auto_mpg_fixed_cleaned.csv")
// // println(auto_mat)

// banner ("auto_mpg (x, y) dataset")
// val (x, y) = auto_tab.toMatrixDD (ArrayBuffer.range (1, 7), 0)
// println (s"x = $x")
// println (s"y = $y")

// val auto_mat = MatrixD.load("auto_mpg_fixed_cleaned.csv")
// println(auto_mat)

// val x = auto_mat(?, 0 to 6)
// val y = auto_mat(?, 7)

// println (x)
// println (y)




//     banner ("Forward Selection Test")
//     val (cols, rSq) = mod.forwardSelAll ()                         // R^2, R^2 Bar, R^2 cv
//     val k = cols.size
//     val t = VectorD.range (1, k)                                   // instance index
//     new PlotM (t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
//                "R^2 vs n for Forward Selection - LassoRegression", lines = true)
//     println (s"rSq = $rSq")

//     banner ("Backward Elimination Test")
//     val (cols2, rSq2) = mod.backwardElimAll ()                       // R^2, R^2 Bar, R^2 cv
//     val k2 = cols2.size
//     val t2 = VectorD.range (1, k2)                                   // instance index
//     new PlotM (t2, rSq2.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
//                "R^2 vs n for Backward Elimination - LassoRegression", lines = true)
//     println (s"rSq = $rSq2")

//     banner ("Stepwise Selection Test")
//     val (cols3, rSq3) = mod.stepRegressionAll ()                     // R^2, R^2 Bar, R^2 cv
//     val k3 = cols3.size
//     val t3 = VectorD.range (1, k3)                                   // instance index
//     new PlotM (t3, rSq3.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
//                "R^2 vs n for Stepwise Selection - LassoRegression", lines = true)
//     println (s"rSq = $rSq3")
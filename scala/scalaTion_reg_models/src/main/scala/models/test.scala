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

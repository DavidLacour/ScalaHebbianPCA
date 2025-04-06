import scala.math._
import scala.collection.mutable.ArrayBuffer

object PCAHebbianComparison {
  def main(args: Array[String]): Unit = {
    val X : Array[Double] = Array(3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8)
    val Y : Array[Double] = Array(3, 4, 3, 5, 4, 5, 6, 7, 6, 8, 7, 8)
    println("Original data shape: " + X.length + " x 2")
    println("Data points:")
    for (i <- X.indices) {
      println(s"Point ${i+1}: (${X(i)}, ${Y(i)})")
    }
    val mean_X = X.sum / X.length
    val mean_Y = Y.sum / Y.length
    println(s"\nMean values: X = $mean_X, Y = $mean_Y")
    val X_centered = X.map(_ - mean_X)
    val Y_centered = Y.map(_ - mean_Y)
    val data_centered = X_centered.zip(Y_centered).map { case (x, y) => Array(x, y) }
    println("\nCentered data:")
    data_centered.foreach(point => println(s"(${point(0)}, ${point(1)})"))

    //PCA:
    // Covariance Matrix
    val n = X.length
    val c_xx = X_centered.map(x => x * x).sum / n
    val c_yy = Y_centered.map(y => y * y).sum / n
    val c_xy = (X_centered zip Y_centered).map { case (x, y) => x * y }.sum / n
    val covMatrix = Array(
      Array(c_xx, c_xy),
      Array(c_xy, c_yy)
    )
    println("\nCovariance Matrix:")
    println(s"[${covMatrix(0)(0)}, ${covMatrix(0)(1)}]")
    println(s"[${covMatrix(1)(0)}, ${covMatrix(1)(1)}]")

    // Calculate eigenvalues and eigenvectors manually
    val trace = c_xx + c_yy
    val det = c_xx * c_yy - c_xy * c_xy
    val discriminant = sqrt(trace * trace - 4 * det)
    val eigenvalue1 = (trace + discriminant) / 2
    val eigenvalue2 = (trace - discriminant) / 2

    val ev1_x = c_xy
    val ev1_y = eigenvalue1 - c_xx
    val ev1_norm = sqrt(ev1_x * ev1_x + ev1_y * ev1_y)
    val eigenvector1 = Array(ev1_x / ev1_norm, ev1_y / ev1_norm)

    val ev2_x = c_xy
    val ev2_y = eigenvalue2 - c_xx
    val ev2_norm = sqrt(ev2_x * ev2_x + ev2_y * ev2_y)
    val eigenvector2 = Array(ev2_x / ev2_norm, ev2_y / ev2_norm)

    println("\nPCA Results:")
    println(s"Principal component: [${eigenvector1(0)}, ${eigenvector1(1)}]")
    println(s"Explained variance ratio: ${eigenvalue1 / (eigenvalue1 + eigenvalue2)}")

    // Project the data onto the principal component
    val data_pca = data_centered.map(point =>
      point(0) * eigenvector1(0) + point(1) * eigenvector1(1)
    )

    val data_projected = data_pca.map(pc =>
      Array(pc * eigenvector1(0) + mean_X, pc * eigenvector1(1) + mean_Y)
    )

    println("\nPCA Projected Data:")
    data_projected.foreach(point => println(s"(${point(0)}, ${point(1)})"))

    // Hebbian Learning:
    // Initialize weights randomly
    val rnd = new scala.util.Random(42)
    var w = Array(rnd.nextDouble(), rnd.nextDouble())
    w = normalizeWeights(w)
    val eta = 0.01  // Learning rate
    val n_iterations = 100
    val weightsHistory = ArrayBuffer[Array[Double]]()
    weightsHistory.append(w.clone())

    for (_ <- 0 until n_iterations) {
      for (i <- data_centered.indices) {
        val x = data_centered(i)
        // Hebbian update rule: Δw = eta * x * (w^T * x)
        val w_dot_x = w(0) * x(0) + w(1) * x(1)
        val delta_w = Array(
          eta * x(0) * w_dot_x,
          eta * x(1) * w_dot_x
        )
        w = Array(w(0) + delta_w(0), w(1) + delta_w(1))
        // Normalize to prevent explosion
        w = normalizeWeights(w)
      }
      weightsHistory.append(w.clone())
    }

    println("\nHebbian Learning Results:")
    println(s"Final weight vector: [${w(0)}, ${w(1)}]")

    // Calculate similarity to PCA principal component
    val similarity = abs(w(0) * eigenvector1(0) + w(1) * eigenvector1(1)) /
      (sqrt(w(0) * w(0) + w(1) * w(1)) * sqrt(eigenvector1(0) * eigenvector1(0) + eigenvector1(1) * eigenvector1(1)))

    println(s"Similarity to PCA principal component: $similarity")

    // Project the data onto the Hebbian weight vector
    val data_hebbian = data_centered.map(point =>
      point(0) * w(0) + point(1) * w(1)
    )

    val data_hebbian_projected = data_hebbian.map(h =>
      Array(h * w(0) + mean_X, h * w(1) + mean_Y)
    )

    println("\nHebbian Projected Data:")
    data_hebbian_projected.foreach(point => println(s"(${point(0)}, ${point(1)})"))

    // Display final results
    println("\nSummary:")
    println(s"PCA principal component: [${eigenvector1(0)}, ${eigenvector1(1)}]")
    println(s"Hebbian final weights: [${w(0)}, ${w(1)}]")
    println(s"Similarity between methods: $similarity")
  }

  /**
   * Normalize a weight vector to have unit length
   */
  def normalizeWeights(w: Array[Double]): Array[Double] = {
    val norm = sqrt(w(0) * w(0) + w(1) * w(1))
    Array(w(0) / norm, w(1) / norm)
  }
}
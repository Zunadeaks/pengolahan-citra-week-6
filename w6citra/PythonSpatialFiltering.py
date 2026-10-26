
import numpy as np
import matplotlib.pyplot as plt


# 1. Manual 2D Convolution dengan Zero-Padding
def convolution_2d(image, kernel):
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    height, width = image.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    # Zero-padding agar ukuran output tetap sama
    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0
    )

    # Membalik kernel untuk konvolusi
    flipped_kernel = np.flip(kernel, axis=(0, 1))

    output = np.zeros((height, width), dtype=np.float64)

    # Perhitungan konvolusi secara manual
    for i in range(height):
        for j in range(width):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * flipped_kernel)

    return output


# 2. Smoothing menggunakan Averaging (Box) Filter
def smoothing_filter(image):
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0

    result = convolution_2d(image, kernel)

    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


# 3. Sharpening menggunakan Laplacian Filter
def sharpening_filter(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float64)

    # Menghitung respons Laplacian
    laplacian = convolution_2d(image, kernel)

    # Menambahkan respons Laplacian ke gambar asli
    result = np.asarray(image, dtype=np.float64) + laplacian

    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


# Menampilkan matriks hasil di terminal
def print_matrix(title, matrix):
    print(f"\n{title}:")
    print(matrix)


# Menampilkan figure perbandingan hasil filter
def show_results(image, convolution, smoothing, sharpening):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    results = [
        (image, "Input Grayscale", 0, 255),
        (convolution, "2D Convolution", None, None),
        (smoothing, "Smoothing - Box Filter", 0, 255),
        (sharpening, "Sharpening - Laplacian", 0, 255)
    ]

    for ax, (data, title, vmin, vmax) in zip(
        axes.flat, results
    ):
        ax.imshow(
            data,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )

        ax.set_title(title)
        ax.set_xlabel("Kolom")
        ax.set_ylabel("Baris")

        # Menampilkan nilai setiap piksel
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]

                if np.issubdtype(data.dtype, np.floating):
                    label = f"{value:.1f}"
                else:
                    label = str(value)

                ax.text(
                    j, i, label,
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=8
                )

    fig.suptitle(
        "Python W6 - Spatial Filtering",
        fontsize=15
    )

    plt.tight_layout()

    # Menyimpan figure ke file PNG
    plt.savefig(
        "spatial_filtering_results.png",
        dpi=300,
        bbox_inches="tight"
    )

    # Menampilkan figure
    plt.show()


# Main function
def main():
    # Contoh gambar grayscale 5x5
    image = np.array([
        [10, 10, 10, 10, 10],
        [10, 50, 50, 50, 10],
        [10, 50, 200, 50, 10],
        [10, 50, 50, 50, 10],
        [10, 10, 10, 10, 10]
    ], dtype=np.uint8)

    # Kernel untuk menguji konvolusi 2D
    test_kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ], dtype=np.float64)

    # Menguji setiap fungsi secara terpisah
    convolution_result = convolution_2d(image, test_kernel)
    smoothing_result = smoothing_filter(image)
    sharpening_result = sharpening_filter(image)

    # Menampilkan hasil di terminal
    print_matrix("Input Image", image)
    print_matrix("2D Convolution", convolution_result)
    print_matrix("Smoothing Result", smoothing_result)
    print_matrix("Sharpening Result", sharpening_result)

    # Memeriksa ukuran output
    print("\nUkuran gambar:")
    print("Input      :", image.shape)
    print("Convolution:", convolution_result.shape)
    print("Smoothing  :", smoothing_result.shape)
    print("Sharpening :", sharpening_result.shape)

    # Menampilkan dan menyimpan figure
    show_results(
        image,
        convolution_result,
        smoothing_result,
        sharpening_result
    )


if __name__ == "__main__":
    main()

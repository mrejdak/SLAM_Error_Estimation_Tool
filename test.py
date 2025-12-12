from mock_data import mock_test_data, load_data_measurements, load_data_csv
from error_estimator import match_and_estimate
from visualize_track_differences import  visualize_matching

def test():
    # positions, colors = mock_test_data(noise=True)
    positions, colors = load_data_csv()
    print(len(positions))
    real_positions, real_colors = load_data_measurements()

    result = match_and_estimate(positions, real_positions, colors, real_colors)

    fig, ax, result, plt = visualize_matching(positions, real_positions, colors, real_colors, result)
    plt.show()
    print(f"RMS: {result['rms']}cm")
    print(f"Residuals: {result['residuals']}")

test()
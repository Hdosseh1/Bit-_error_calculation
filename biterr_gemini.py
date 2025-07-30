import numpy as np
import matplotlib.pyplot as plt

def calculate_ber_bpsk_awgn(num_bits, eb_no_db_range):
    """
    Calculates the Bit Error Rate (BER) for BPSK modulation in an AWGN channel.

    Args:
        num_bits (int): The total number of bits to simulate.
        eb_no_db_range (list or range): A list or range of Eb/No values in dB.

    Returns:
        list: A list containing the calculated BER for each Eb/No value.
    """
    ber_results = []

    for eb_no_db in eb_no_db_range:
        # Convert Eb/No from dB to linear scale
        eb_no_linear = 10.0**(eb_no_db / 10.0)

        # Generate random BPSK symbols (-1 or 1)
        transmitted_bits = 2 * (np.random.rand(num_bits) >= 0.5) - 1

        # Calculate noise standard deviation based on Eb/No
        # For BPSK, Es/No = Eb/No, and noise power is 1.
        noise_std = 1 / np.sqrt(2 * eb_no_linear)

        # Add AWGN noise to the transmitted signal
        received_signal = transmitted_bits + noise_std * np.random.randn(num_bits)

        # Demodulate the received signal (decision based on sign)
        demodulated_bits = 2 * (received_signal >= 0) - 1

        # Count errors
        errors = (transmitted_bits != demodulated_bits).sum()

        # Calculate BER
        ber = errors / num_bits
        ber_results.append(ber)

        print(f"Eb/No (dB): {eb_no_db}, Errors: {errors}, BER: {ber:.6f}")

    return ber_results

# Simulation parameters
num_bits_to_simulate = 1000000  # Number of bits for simulation
eb_no_db_values = range(0, 11)  # Eb/No range from 0 to 10 dB

# Calculate BER
calculated_ber = calculate_ber_bpsk_awgn(num_bits_to_simulate, eb_no_db_values)

# Plotting the results
plt.figure()
plt.plot(eb_no_db_values, calculated_ber, 'bo-', label='Simulated BER')
plt.xlabel('Eb/No (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.yscale('log')  # Use logarithmic scale for BER
plt.grid(True)
plt.title('BER vs. Eb/No for BPSK in AWGN Channel')
plt.legend()
plt.show()
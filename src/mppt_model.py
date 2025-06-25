def mppt_algorithm(voltage, current):
    """
    Simple Perturb and Observe (P&O) MPPT algorithm.
    """
    dV = 0.1  # Voltage perturbation step

    # Previous values for the P&O algorithm
    prev_voltage = voltage
    prev_current = current
    prev_power = prev_voltage * prev_current

    # Simulate a change in voltage (perturbation)
    new_voltage = voltage + dV
    new_current = current  # In a real implementation, you'd measure this

    # Calculate new power
    new_power = new_voltage * new_current

    # If power increases, continue in the same direction
    if new_power > prev_power:
        optimized_voltage = new_voltage
    else:
        # If power decreases, reverse direction
        optimized_voltage = prev_voltage - dV

    return optimized_voltage

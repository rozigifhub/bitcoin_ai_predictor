import numpy as np


class LSTMNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        input_size  : jumlah fitur input per time-step(oclv = 5 input)
        hidden_size : ukuran hidden state
        output_size : jumlah output (untuk prediksi 1 angka = 1)
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # WEIGHT FOR FORGET GATE
        self.weight_forget_gate_input = np.random.randn(hidden_size, input_size) * 0.01
        self.weight_forget_gate_hidden = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bias_forget_gate = np.zeros((hidden_size, 1))

        # WEIGHT FOR INPUT GATE
        self.weight_input_gate_input = np.random.randn(hidden_size, input_size) * 0.01
        self.weight_input_gate_hidden = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bias_input_gate = np.zeros((hidden_size, 1))

        # WEIGHT FOR CANDIDATE STATE GATE
        self.weight_candidate_gate_input = np.random.randn(hidden_size, input_size) * 0.01
        self.weight_candidate_gate_hidden = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bias_candidate_gate = np.zeros((hidden_size, 1))

        # WEIGHT FOR OUTPUT GATE
        self.weight_output_gate_input = np.random.randn(hidden_size, input_size) * 0.01
        self.weight_output_gate_hidden = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bias_output_gate = np.zeros((hidden_size, 1))

        # WEIGHT FOR FINAL OUTPUT LAYER
        self.weight_output_layer = np.random.randn(output_size, hidden_size) * 0.01
        self.bias_output_layer = np.zeros((output_size, 1))


    # Activation Functions

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    # Forward Pass LSTM

        """
        input_sequence: array bentuk (sequence_length, input_size)
        semua nilai intermediate disimpan untuk backprop nanti
        """

        sequence_length = len(input_sequence)

        # Simpan semua gate dan state untuk debugging/backprop
        self.forget_gate_values = []
        self.input_gate_values = []
        self.candidate_gate_values = []
        self.output_gate_values = []

        self.cell_state_history = []
        self.hidden_state_history = []

        # Inisialisasi cell state & hidden state
        hidden_state_previous = np.zeros((self.hidden_size, 1))
        cell_state_previous = np.zeros((self.hidden_size, 1))

        # Iterasi tiap time-step
        for t in range(sequence_length):
            current_input = input_sequence[t].reshape(-1, 1)

            # -----------------------
            # Compute Forget Gate
            # -----------------------
            forget_gate = self.sigmoid(
                self.weight_forget_gate_input @ current_input +
                self.weight_forget_gate_hidden @ hidden_state_previous +
                self.bias_forget_gate
            )

            # -----------------------
            # Compute Input Gate
            # -----------------------
            input_gate = self.sigmoid(
                self.weight_input_gate_input @ current_input +
                self.weight_input_gate_hidden @ hidden_state_previous +
                self.bias_input_gate
            )

            # -------------------------------
            # Compute Candidate Cell State Gate
            # -------------------------------
            candidate_cell_state = self.tanh(
                self.weight_candidate_gate_input @ current_input +
                self.weight_candidate_gate_hidden @ hidden_state_previous +
                self.bias_candidate_gate
            )

            # -----------------------
            # Compute Output Gate
            # -----------------------
            output_gate = self.sigmoid(
                self.weight_output_gate_input @ current_input +
                self.weight_output_gate_hidden @ hidden_state_previous +
                self.bias_output_gate
            )

            # -----------------------
            # Update Cell State
            # -----------------------
            cell_state_current = (
                forget_gate * cell_state_previous +
                input_gate * candidate_cell_state
            )

            # -----------------------
            # Update Hidden State
            # -----------------------
            hidden_state_current = output_gate * self.tanh(cell_state_current)

            # Simpan untuk backprop
            self.forget_gate_values.append(forget_gate)
            self.input_gate_values.append(input_gate)
            self.candidate_gate_values.append(candidate_cell_state)
            self.output_gate_values.append(output_gate)

            self.cell_state_history.append(cell_state_current)
            self.hidden_state_history.append(hidden_state_current)

            # Update previous states
            hidden_state_previous = hidden_state_current
            cell_state_previous = cell_state_current

        # -----------------------
        # Final Output Layer
        # -----------------------
        final_output = (
            self.weight_output_layer @ hidden_state_current +
            self.bias_output_layer
        )

        return final_output, hidden_state_current

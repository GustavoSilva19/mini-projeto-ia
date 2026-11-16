
import numpy as np


class MLP:

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        

        self.parameter_count = (
            input_size * hidden_size
            + hidden_size
            + hidden_size * self.output_size
            + self.output_size
        )

    @staticmethod
    def sigmoid(x):

        x = np.clip(x, -500, 500)

        return 1.0 / (1.0 + np.exp(-x))

    def unpack_parameters(self, parameters):

        expected_size = self.parameter_count

        if len(parameters) != expected_size:
            raise ValueError(
                f"Esperados {expected_size} parâmetros, "
                f"recebidos {len(parameters)}."
            )

        position = 0

        
        end = position + self.input_size * self.hidden_size

        weights_input_hidden = parameters[position:end].reshape(
            self.input_size,
            self.hidden_size
        )

        position = end

        
        end = position + self.hidden_size

        bias_hidden = parameters[position:end]

        position = end

        
        end = position + self.hidden_size * self.output_size

        weights_hidden_output = parameters[position:end].reshape(
            self.hidden_size,
            self.output_size
        )

        position = end

        
        bias_output = parameters[position:]

        return (
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output
        )

    def predict_proba(self, X, parameters):
    

        (
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output
        ) = self.unpack_parameters(parameters)

        
        hidden_input = (
            np.dot(X, weights_input_hidden)
            - bias_hidden
        )

        hidden_output = self.sigmoid(hidden_input)

        
        output_input = (
            np.dot(hidden_output, weights_hidden_output)
            - bias_output
        )

        output = self.sigmoid(output_input)

        return output.ravel()

    def predict(self, X, parameters, threshold=0.5):
        

        probabilities = self.predict_proba(X, parameters)

        return (probabilities >= threshold).astype(int)

    def calculate_mse(self, X, y, parameters):
        
        predictions = self.predict_proba(X, parameters)

        y = np.asarray(y, dtype=float).ravel()

        return float(np.mean((y - predictions) ** 2))

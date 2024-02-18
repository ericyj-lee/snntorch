import torch
from snntorch.spike import SpikeFunctionBase
from snntorch import functional_surrogate

class LIF_mutiple(Leaky):
    def __init__(
        self,
        beta,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        super().__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
            reset_delay,
        )

    def _base_state_function(self, input_):
        base_fn = self.beta.clamp(0, 1) * self.mem + input_
        return base_fn

    def _base_sub(self, input_):
        return self._base_state_function(input_) - self.reset * self.threshold

    def _base_zero(self, input_):
        self.mem = (1 - self.reset) * self.mem
        return self._base_state_function(input_)

    def _base_int(self, input_):
        return self._base_state_function(input_)

    def forward(self, input_, mem=None):
        if not mem == None:
            self.mem = mem
        
        if self.init_hidden and not mem == None:
            raise TypeError(
                "`mem` should not be passed as an argument while `init_hidden=True`"
            )

        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)

        self.reset = self.mem_reset(self.mem)
        self.mem = self.state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        if self.inhibition:
            spk = self.fire_inhibition(self.mem.size(0), self.mem)  # batch_size
        else:
            spk = self.fire(self.mem)

        if not self.reset_delay:
            do_reset = spk / self.graded_spikes_factor - self.reset  # avoid double reset
            if self.reset_mechanism_val == 0:  # reset by subtraction
                self.mem = self.mem - do_reset * self.threshold
            elif self.reset_mechanism_val == 1:  # reset to zero
                self.mem = self.mem - do_reset * self.mem

        # Check for first-order spike condition
        if self.mem > self.n * self.threshold:
            n_spikes = int(self.mem / self.threshold)
            spk = spk * n_spikes
            if self.reset_mechanism == "subtract":
                self.mem = self.mem - n_spikes * self.threshold

        if self.output:
            return spk, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.mem

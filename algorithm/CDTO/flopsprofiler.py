from paleo.profilers.flops_profiler import FlopsProfiler as PaleoFlopsProfiler
from paleo.profilers.base import ProfilerOptions


class FlopsProfiler:
    @staticmethod
    def profile(layer_spec):
        layer = layer_spec.operation

        assert layer is not None, f'{layer_spec} has no operation'


        profiler_options = ProfilerOptions()
        '''

        CODE OF ProfilerOptions() @ CDTO/code/paleo/profilers/base.py
        class ProfilerOptions(object):
        """The options for profilers"""

        def __init__(self):
            self.direction = 'forward'  # forward, backward
            self.gradient_wrt = 'data'  # data, filter, None
            self.num_warmup = 10
            self.num_iter = 50
            self.use_cudnn_heuristics = False

            # By default we don't include bias and activation.
            # this will make layer-wise comparison easier.
            self.include_bias_and_activation = False

            # Platform percent of peek.
            self.ppp_comp = 1.0
            self.ppp_comm = 1.0
        '''
        direction = 'forward'
        profiler_options.direction = direction
        profiler_options.use_cudnn_heuristics = False
        profiler_options.include_bias_and_activation = False
        class device:
            def __init__(self,type):
                self.type = type
            def is_gpu(self):
                return self.type == 'cpu'
        profiler = PaleoFlopsProfiler(profiler_options, device('cpu'))

        '''
        class FlopsProfiler(BaseProfiler):
            # from paleo.profilers.flops_profiler import FlopsProfiler as PaleoFlopsProfiler
            def __init__(self, options, device):
                super(FlopsProfiler, self).__init__('FlopsProfiler', options)
                self._device = device
                if not self._device.is_gpu:
                    self.options.use_cudnn_heuristics = False

                if self.options.use_cudnn_heuristics:
                    from paleo.profilers import cudnn_profiler as cudnn
                    self.cudnn = cudnn

                self.estimate_remote_fetch = self._estimate_remote_fetch
                self.estimate_comm_time = self._estimate_comm_time
                self.cpu1estimate_comm_time = self._cpu1estimate_comm_time
                self.cpu2estimate_comm_time = self._cpu2estimate_comm_time
                self.cpu3estimate_comm_time = self._cpu3estimate_comm_time

        '''
        gflops = profiler.flop_profile(layer)
        # print('time',time)

        return gflops



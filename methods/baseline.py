class Method:
    def __init__(self, **kwargs):
        pass

    def before_sampling(self, pipe, meta):
        pass

    def on_step(self, *args, **kwargs):
        pass

    def after_sampling(self, pipe, meta, extra):
        pass


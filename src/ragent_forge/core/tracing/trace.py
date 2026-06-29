from ragent_forge.app.models import RagTrace, TraceStep


def add_step(trace: RagTrace, step: TraceStep) -> RagTrace:
    return trace.model_copy(update={"steps": [*trace.steps, step]})

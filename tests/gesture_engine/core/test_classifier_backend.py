import app.gesture_engine.core.classifier_backend as cb


def make_sequence(t: int = 10, points: int = 21):
    # tworzy sekwencje T ramek, kazda 21x3 float
    frame = [(0.0, 0.0, 0.0) for _ in range(points)]
    return [frame for _ in range(t)]


def test_torch_gru_backend_load_and_predict_placeholder():
    backend = cb.TorchGRUBackend(device=None, time_window=10)
    backend.load("/path/to/dummy.pt")

    seq = make_sequence(10)
    pred = backend.predict(seq)

    assert pred.label is None
    assert pred.confidence == 0.0
    assert isinstance(pred.meta, dict)


def test_torch_gru_backend_predict_requires_load():
    backend = cb.TorchGRUBackend()
    # bez load powinien rzucic
    try:
        backend.predict(make_sequence(10))
        assert False, "predict powinien rzucic bez load()"
    except RuntimeError:
        pass


def test_torch_gru_backend_close_noop():
    backend = cb.TorchGRUBackend()
    assert backend.close() is None

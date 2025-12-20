from app.sign_language.early_stopping import EarlyStopping


def test_early_stopping_init():
    es = EarlyStopping(patience=5, delta=0.01, verbose=False)

    assert es.patience == 5
    assert es.delta == 0.01
    assert es.counter == 0
    assert es.best_score is None
    assert es.early_stop is False


def test_early_stopping_first_call():
    es = EarlyStopping(patience=5, delta=0.01, verbose=False)

    should_stop = es(val_loss=1.0)

    assert should_stop is False
    assert es.best_score == -1.0
    assert es.counter == 0


def test_early_stopping_improvement():
    es = EarlyStopping(patience=3, delta=0.01, verbose=False)

    es(val_loss=1.0)
    should_stop = es(val_loss=0.8)  # poprawa

    assert should_stop is False
    assert es.counter == 0
    assert es.best_score == -0.8


def test_early_stopping_no_improvement():
    es = EarlyStopping(patience=3, delta=0.01, verbose=False)

    es(val_loss=1.0)
    es(val_loss=1.0)  # brak poprawy

    assert es.counter == 1
    assert es.early_stop is False


def test_early_stopping_triggers():
    es = EarlyStopping(patience=3, delta=0.01, verbose=False)

    es(val_loss=1.0)
    es(val_loss=1.0)  # +1
    es(val_loss=1.0)  # +2
    should_stop = es(val_loss=1.0)  # +3 -> trigger

    assert should_stop is True
    assert es.early_stop is True
    assert es.counter == 3


def test_early_stopping_resets_counter_on_improvement():
    es = EarlyStopping(patience=3, delta=0.01, verbose=False)

    es(val_loss=1.0)
    es(val_loss=1.0)  # +1
    es(val_loss=0.8)  # poprawa -> reset
    es(val_loss=0.8)  # +1

    assert es.counter == 1
    assert es.early_stop is False


def test_early_stopping_respects_delta():
    es = EarlyStopping(patience=2, delta=0.1, verbose=False)

    es(val_loss=1.0)
    es(val_loss=0.95)  # poprawa mniejsza niz delta -> brak uznania za poprawe

    assert es.counter == 1

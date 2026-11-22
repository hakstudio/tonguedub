import pytest

from tonguedub import config as config_mod


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Testler geliştiricinin gerçek .env/config.toml dosyalarını ve ortam değişkenlerini görmesin, onlara yazmasın."""
    monkeypatch.setattr(config_mod, "env_files", lambda: [])
    monkeypatch.setattr(config_mod, "USER_CONFIG", config_mod.Path("/nonexistent/tonguedub/config.toml"))
    monkeypatch.setattr(config_mod, "project_root", lambda: None)
    for var in (config_mod.ENV_CONFIG, config_mod.ENV_API_KEY):
        monkeypatch.setenv(var, "placeholder")  # kayıt altına al: test sonunda eski (yok) hâline döner
        monkeypatch.delenv(var)

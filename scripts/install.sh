#!/bin/zsh
# Tonguedub kurulumu (Apple Silicon Mac). Depo kökünden ya da herhangi bir yerden çalıştırılabilir:
#   ./scripts/install.sh
# Tekrar çalıştırmak güvenlidir: mevcut config.toml ve kısayol (aynı yolu gösteriyorsa) korunur.
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Tonguedub yalnızca Apple Silicon Mac'te test edildi (bulunan: $(uname -s)/$(uname -m))." >&2
  exit 1
fi
command -v brew >/dev/null || { echo "Homebrew gerekli: https://brew.sh" >&2; exit 1; }

# 1) sistem araçları (yalnızca eksik olanlar)
for pkg in uv ffmpeg whisperkit-cli rubberband; do
  if ! brew list --formula "$pkg" >/dev/null 2>&1; then
    echo "→ brew install $pkg"
    brew install "$pkg"
  fi
done

# 2) Python ortamı
[[ -x .venv/bin/python ]] || uv venv --python 3.11 .venv
echo "→ paketler kuruluyor (ilk seferde birkaç dakika sürer)"
uv pip install --python .venv/bin/python -e ".[dev]"
# chatterbox-tts: `pkuseg` bağımlılığı derlenmediği için --no-deps; geri kalan bağımlılıklar pyproject.toml'da
uv pip install --python .venv/bin/python --no-deps chatterbox-tts

# 3) ayar dosyası
if [[ ! -f config.toml ]]; then
  cp config.example.toml config.toml
  echo "→ config.toml oluşturuldu: (varsa) Whisper model yolunu düzenle."
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
  echo "→ .env oluşturuldu: DEEPSEEK_API_KEY değerini içine yaz (dosya yalnızca sana açık)."
fi

# 4) komut kısayolu (venv'i etkinleştirmeden `tonguedub` çalışsın; bu klasörün yolunu içerir)
mkdir -p "$HOME/.local/bin"
SHIM="$HOME/.local/bin/tonguedub"
cat > "$SHIM" <<EOS
#!/bin/zsh
# Tonguedub kısayolu (scripts/install.sh üretti). Uzun işlerde Mac uyumasın diye caffeinate kullanır.
exec caffeinate -i "$ROOT/.venv/bin/python" -m tonguedub "\$@"
EOS
chmod +x "$SHIM"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "! ~/.local/bin PATH'te değil; zsh için: echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc" ;; esac

echo
echo "Kurulum bitti. Kontrol için: tonguedub doctor"

"""Ölçümle belirlenmiş sabitler (gerekçeler CLAUDE.md'de). Kullanıcı ayarı değildir; ayarlar config.toml'dadır."""
import math

SR = 44100                      # iç çalışma örnekleme hızı
TR_CHARS_PER_SEC = 15.0         # çeviri için hedef konuşma hızı (karakter/sn)
CHARS_PER_SEC = 14.0            # grup süresi tahmini (karakter/sn)
TTS_SYL_RATE = 4.5              # Chatterbox Türkçe doğal hızı (hece/sn)
VOCAL_MIN_RMS = 10 ** (-52 / 20)  # vokal kanalında bundan sessizse segment "konuşma değil" sayılır
LOW_HZ = 150.0                  # bunun altındaki medyan perde "alçak perde grubu"
SHORT_SEC = 2.5                 # bundan kısa satırlar birleştirmeye uygun
MERGE_CAP = 4                   # grup başına en fazla satır (uzun gruplarda sessizlikten kesme güvenilmez)
REROLL_ST = 3.5                 # klip perdesi hedeften bu kadar yarım tondan fazla saparsa yeniden üret

MAX_PUSH = 0.4                  # sığmayan cümlenin ardındakini en fazla bu kadar ertele (sn)
MAX_RATE = 1.5                  # sığdırmak için en fazla hızlandırma
SOFT_RATE = 1.25                # bu orana kadar yalnızca hızlandırarak sığdır
GAP = 0.04                      # klipler arası asgari boşluk (sn)
TRUNC_FADE = 0.08               # kesilen klibin fade-out süresi (sn)

PITCH_DEADBAND = 2.0            # yarım ton: bu kadar sapma doğal çeşitlilik sayılır
PITCH_MAX_FIX = 4.5             # bundan büyük sapma düzeltilmez (işaretlenir)
EXPRESS_MAX = 4.0               # takip edilen perde sapması en fazla bu kadar yarım ton

assert math.isclose(SR, 44100)

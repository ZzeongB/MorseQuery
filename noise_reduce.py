import noisereduce as nr
from scipy.io import wavfile

rate, data = wavfile.read(
    "live/logs/sessions/20260309_095222_c6-AyP7vAeehhCXxAAAD/audio/20260309_095543_c6-AyP7vAeehhCXxAAAD_sum0.wav"
)
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)

# install ffmpeg with svt-av1

# Reference: 
# 1. https://osanshouo.github.io/blog/2021/06/17-ffmpeg/
# 2. https://forums.linuxmint.com/viewtopic.php?t=435381

# mkdir
mkdir ~/{.ffmpeg,.ffmpeg-build,.ffmpeg-src}

# Install dependencies
apt install -y \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  libass-dev \
  libfreetype6-dev \
  libgnutls28-dev \
  libtool \
  libvorbis-dev \
  meson \
  ninja-build \
  pkg-config \
  texinfo \
  wget \
  yasm \
  zlib1g-dev \
  openssl
  
# Install library relating to encoding/decoding
apt install -y \
  nasm \
  libx264-dev \
  libx265-dev libnuma-dev \
  libvpx-dev \
  libmp3lame-dev \
  libopus-dev \
  libaom-dev 

# Install svt-av1
cd ~/.ffmpeg-src && \
wget https://gitlab.com/AOMediaCodec/SVT-AV1/-/archive/v2.3.0/SVT-AV1-v2.3.0.tar.gz && \
tar xf SVT-AV1-v2.3.0.tar.gz && \
mv SVT-AV1-v2.3.0 SVT-AV1 && \
cd SVT-AV1/Build && \
cmake -G "Unix Makefiles" \
  -DCMAKE_INSTALL_PREFIX="$HOME/.ffmpeg-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_DEC=OFF \
  -DBUILD_SHARED_LIBS=OFF .. && \
make -j4 && \
make install

# Build ffmpeg
cd ~/.ffmpeg-src && \
wget https://ffmpeg.org/releases/ffmpeg-7.1.tar.gz && \
tar xf ffmpeg-7.1.tar.gz && \
mv ffmpeg-7.1 ffmpeg && \
cd ffmpeg && \
PKG_CONFIG_PATH="$HOME/.ffmpeg-build/lib/pkgconfig" ./configure \
  --prefix="$HOME/.ffmpeg-build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/.ffmpeg-build/include" \
  --extra-ldflags="-L$HOME/.ffmpeg-build/lib" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/.ffmpeg/bin" \
  --disable-ffplay \
  --enable-gpl \
  --enable-gnutls \
  --enable-libaom \
  --enable-libass \
  --enable-libfreetype \
  --enable-libmp3lame \
  --enable-libopus \
  --enable-libsvtav1 \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree && \
make -j8 && \
make install 

# Add ffmpeg to PATH
echo 'export PATH="$HOME/.ffmpeg/bin:$PATH"' >> ~/.bashrc
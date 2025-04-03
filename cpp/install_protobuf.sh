git clone --progress -b v3.13.0 https://github.com/protocolbuffers/protobuf && \
    ( \
      cd protobuf; \
      mkdir protobuf_build; \
      cd protobuf_build; \
      cmake ../cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -Dprotobuf_BUILD_SHARED_LIBS=ON \
        -Dprotobuf_BUILD_TESTS=OFF; \
      make -j4 install; \
    ) && \
    rm -rf protobuf

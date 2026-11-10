# syntax=docker/dockerfile:1.6
#
# snicholls-core - the Ubuntu build and test image.
#
# Ubuntu is a supported target, and this is how it is tested locally, the same
# way CI tests it: scripts/test_ubuntu.sh builds this image and runs every
# CTest under GCC and Clang, Release and Debug, plus the sanitizers.
#
# Core's own image, deliberately small: core depends on nothing, so the image
# holds only compilers and build tools, plus compiler-rt so that Clang's
# sanitizers link.

FROM ubuntu:24.04

LABEL org.opencontainers.image.title="snicholls-core Ubuntu builder" \
      org.opencontainers.image.description="Ubuntu 24.04: GCC 13, Clang 18 with compiler-rt, CMake, Ninja, Doxygen"

ENV DEBIAN_FRONTEND=noninteractive

# libclang-rt-18-dev is compiler-rt: without it -fsanitize=address,undefined,thread
# compiles with Clang and then fails to link.
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential g++-13 \
      clang-18 libclang-rt-18-dev \
      cmake ninja-build \
      python3-minimal git ca-certificates curl && \
    ln -sf /usr/bin/clang-18 /usr/local/bin/clang && \
    ln -sf /usr/bin/clang++-18 /usr/local/bin/clang++ && \
    rm -rf /var/lib/apt/lists/*

# Doxygen, the same version as on macOS. Ubuntu 24.04's package
# is 1.9.8, older than the docs build is written for, so the official x86-64
# release binary is installed, pinned by version and SHA-256. There is no ARM64
# release binary: on an ARM host the apt package is installed instead and
# scripts/test_ubuntu.sh skips the docs check there, saying so.
ARG DOXYGEN_VERSION=1.18.0
ARG DOXYGEN_TAG=Release_1_18_0
ARG DOXYGEN_SHA256=14fa81bdc34171edb5f1f02b1d60e74802f0439b77fa44e592565d517d72df90
ARG TARGETARCH
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    if [ "${TARGETARCH:-amd64}" = "amd64" ]; then \
      curl -fsSL -o /tmp/doxygen.tgz \
        "https://github.com/doxygen/doxygen/releases/download/${DOXYGEN_TAG}/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz" && \
      echo "${DOXYGEN_SHA256}  /tmp/doxygen.tgz" | sha256sum -c - && \
      tar -xzf /tmp/doxygen.tgz -C /tmp && \
      install -m 0755 "/tmp/doxygen-${DOXYGEN_VERSION}/bin/doxygen" /usr/local/bin/doxygen && \
      rm -rf /tmp/doxygen.tgz "/tmp/doxygen-${DOXYGEN_VERSION}"; \
    else \
      apt-get update && apt-get install -y --no-install-recommends doxygen && rm -rf /var/lib/apt/lists/*; \
    fi && \
    doxygen --version

# Sources are mounted read-only by scripts/test_ubuntu.sh and copied in, so a
# build inside the container never writes into the checkout.
WORKDIR /work
CMD ["bash"]

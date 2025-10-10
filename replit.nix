{ pkgs }: {
  deps = [
    pkgs.python310Full
    pkgs.pkg-config
    pkgs.zlib
    pkgs.libjpeg_turbo
    pkgs.poppler   # for pdftoppm
    pkgs.tesseract # for OCR
    pkgs.opencv    # for image processing
  ];
  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.libjpeg_turbo
      pkgs.zlib
      pkgs.poppler
      pkgs.opencv
    ];
  };
  shellHook = ''
    python -m pip install --upgrade pip
    pip install -r requirements.txt
  '';
}
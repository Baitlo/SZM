#pragma once

#include <filesystem>
#include <iostream>

class ImageCompressor {
  public:
    ImageCompressor(const char* input_file_name);

    void compress();
    void copy_and_output_file();

    // void compress_2();

  private:
    std::filesystem::path base_path = "/tmp/compress";
    std::filesystem::path input_file_path;
    std::filesystem::path compres_output_file_path;
    std::filesystem::path restore_output_file_path;
    std::filesystem::path text_output_file_path;
};
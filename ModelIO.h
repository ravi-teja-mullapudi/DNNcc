#include <string>
#include <map>
#include "NDArray.h"

typedef std::map<std::string, std::vector<NDArray<float>>> Params;

void save_model_to_disk(std::string model_path, Params &params);
void load_model_from_disk(std::string model_path, Params &params);

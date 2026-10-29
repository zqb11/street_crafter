#include <limits>
#include <adc_map_hub.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>

// Offline map-hub adapter: GSLXYZ01, uint64 count, count * float32 XYZ.
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: dump_lidar_elm INPUT.elm OUTPUT.bin\n";
        return 2;
    }
    std::unique_ptr<adc::map_hub::iMapHubLoader> loader(
        adc::map_hub::iMapHubLoader::createMapHubLoader());
    if (!loader || loader->init(argv[1]) != adc::map_hub::MAP_HUB_OK) return 1;
    const auto* points = loader->getAllMapPoints();
    if (!points) return 1;
    std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
    const char magic[8] = {'G','S','L','X','Y','Z','0','1'};
    const std::uint64_t count = points->size();
    out.write(magic, 8);
    out.write(reinterpret_cast<const char*>(&count), 8);
    for (const auto* p : *points) {
        if (!p) return 1;
        const float xyz[3] = {p->point.x, p->point.y, p->point.z};
        out.write(reinterpret_cast<const char*>(xyz), sizeof(xyz));
    }
    return out ? 0 : 1;
}

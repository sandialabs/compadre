#include "Compadre_KokkosParser.hpp"

using namespace Compadre;

// for InitArguments, pass them directly in to Kokkos
KokkosParser::KokkosParser(Kokkos::InitArguments args, bool print_status) {
    this->ksg = new Kokkos::ScopeGuard(args);
    if (print_status) this->status();
}

// for command line arguments, pass them directly in to Kokkos
KokkosParser::KokkosParser(int narg, char* args[], bool print_status) {
    this->ksg = new Kokkos::ScopeGuard(narg, args);
    if (print_status) this->status();
}

KokkosParser::KokkosParser(std::vector<std::string> stdvec_args, bool print_status) {
    std::vector<char*> char_args;
    for (const auto& arg : stdvec_args) {
        char_args.push_back((char*)arg.data());
    }
    char_args.push_back(nullptr);
    int narg = (int)stdvec_args.size();

    this->ksg = new Kokkos::ScopeGuard(narg, char_args.data());
    if (print_status) this->status();
}

KokkosParser::KokkosParser(bool print_status) : KokkosParser(Kokkos::InitArguments(), print_status) {}

std::string KokkosParser::status() {
    std::stringstream stream;
    Kokkos::print_configuration(stream, true);
    std::string status = stream.str();
    return status;
}

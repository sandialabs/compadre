#include "Compadre_CSVUtilities.hpp"
#include "csv.h"

int main() {

//     std::string line1("0.1,  0.4, 5.32589, A, 2, 2.Bc5o0");
//     
//     std::cout << "CSVHelper tests -------------------------" << std::endl;
//     Compadre::CSVHelper helper;
//     
//     std::vector<std::string> splitLine = helper.splitString(line1);
//     
//     for (int i = 0; i < splitLine.size(); ++i)
//         std::cout << splitLine[i] << std::endl;
//     std::cout << line1 << std::endl;
//     std::cout << "size of splitLine = " << splitLine.size() << std::endl;
//     
//     std::cout << "-------------------------" << std::endl;
//     
//     std::vector<std::string> nospcs;
//     for (auto& s : splitLine)
//         nospcs.push_back(helper.removeSpaces(s));
//     
//     for (int i = 0; i < nospcs.size(); ++i)
//         std::cout << nospcs[i] << std::endl;
//         
//     std::cout << "CSVRow tests-------------------------" << std::endl;
    
    const std::string fname = "testCSV.csv";
//     std::ifstream file(fname);
//     Compadre::CSVRow row;
//     
//     while (file >> row)
//         std::cout << row;
//         
//     std::cout << "-------------------------" << std::endl;
//     
//     std::cout << "CSVFile tests-------------------------" << std::endl;
//     
//     Compadre::CSVFile cFile(fname);
//     std::vector<std::string> cols = cFile.getColumnNames();
//     std::cout << "found columns: " << std::endl;
//     for (auto& col : cols)
//         std::cout << "\t" << col << std::endl;
//         
//     std::vector<std::string> wantCols = {"x", "y", "z", "area", "source3"};
//     
//     cFile.setColumnsToRead(wantCols);
//     
//     std::vector<Compadre::local_index_type> inds = cFile.readColIndices();
//     std::cout << "Read columns ";
//     for (auto& ind : inds)
//         std::cout << ind << ", ";
//     std::cout << std::endl;
    
std::cout << "-------------------------" << std::endl;
std::cout << "-------------------------" << std::endl;
std::cout << "-------------------------" << std::endl;


    int nn;
    io::CSVReader<1> nreader(fname);
    nreader.next_line();
    nreader.read_row(nn);
    std::cout << "file contains " << nn << " particles.\n";

    io::CSVReader<4, io::trim_chars<' '>, io::no_quote_escape<','>, io::throw_on_overflow, 
        io::single_line_comment<'#'>> in(fname);
    std::vector<double> xx, yy, area;
    std::vector<int> boundary_id;
    
    double x = -1000.0;
    double y;
    double a = -1.0;
    int bid = 39;
    in.read_header(io::ignore_extra_column, "x", "y", "area", "boundary_id");
    in.next_line();
    
    while (in.read_row(x, y, a, bid)) {
        xx.push_back(x);
        yy.push_back(y);
        area.push_back(a);
        boundary_id.push_back(bid);
    }
    
    std::cout << "x = " << std::endl;
    for (int i = 0; i < xx.size(); ++i)
        std::cout << "\t" << xx[i] << std::endl;
    
    std::cout << "area = " << std::endl;
    for (int i = 0; i < area.size(); ++i)
        std::cout << "\t" << area[i] << std::endl;
    
    std::cout << "boundary_id = " << std::endl;
    for (int i = 0; i < boundary_id.size(); ++i)
        std::cout << "\t" << boundary_id[i] << std::endl;
return 0;
}

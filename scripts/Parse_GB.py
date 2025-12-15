# recurrent neural network for sequence classificaition
#

from Bio import SeqIO
import sys
import os


def get_taxonomy_string(x):
    """Get's you the whole taxonomy string with accession and species name"""
    return ";".join(
        x.annotations["accessions"]
        + x.annotations["taxonomy"]
        + [x.annotations["organism"]]
    )


def getdata_gb(input_file):
    data = []
    for record in SeqIO.parse(input_file, "gb"):
        taxid = get_taxonomy_string(record)
        seq = str(record.seq)
        data.append([taxid, seq])
    return data


def write_fasta(input_file, output_file):
    with open(output_file, "w") as o:
        for record in SeqIO.parse(input_file, "gb"):
            taxid = get_taxonomy_string(record)
            seq = str(record.seq)
            o.write(">" + taxid + "\n" + seq + "\n")


genbank_file = sys.argv[1]
outfile = sys.argv[2]

write_fasta(genbank_file, outfile)

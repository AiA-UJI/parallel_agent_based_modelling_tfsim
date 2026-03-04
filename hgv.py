import gzip
import xml.etree.ElementTree as ET

FILE = "trafficspeed.xml.gz"

def load_xml(file):
    if file.endswith(".gz"):
        with gzip.open(file, "rb") as f:
            return ET.parse(f)
    else:
        return ET.parse(file)

tree = load_xml(FILE)
root = tree.getroot()

ns = {
    "d": "http://datex2.eu/schema/2/2_0",
    "s": "http://schemas.xmlsoap.org/soap/envelope/"
}

logical = root.find(".//s:Body/d:d2LogicalModel", ns)
payload = logical.find("d:payloadPublication", ns)

measurements = payload.findall(".//d:siteMeasurements", ns)

print("Mostrando PRIMER sensor completo:")
print("----------------------------------")
print(ET.tostring(measurements[0], encoding="unicode", method="xml"))

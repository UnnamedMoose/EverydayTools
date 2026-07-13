
from xml.etree import ElementTree


def set_xml_elem(elem, txt, attrs=None):
    elem.text = txt
    if attrs is not None:
        for attr_key, attr_value in attrs.items():
            elem.set(attr_key, attr_value)

    return elem


def update_nested_xml(root, tag, updates):
    """
    Update a parent XML element and its nested elements based on a dictionary of values.
    Optionally rename the parent element.
    :param root: The root element of the XML.
    :param tag: The current tag name of the parent element to update.
    :param updates: Dictionary with tag names as keys and (text, attributes) tuples as values.
                    The parent tag updates should be included under the key 'self'.
    """

    parent = root.find(f".//{tag}")

    if parent is not None:
        parent.tag = updates["bcType"]

        # Update the nested elements
        for subtag in updates:
            if subtag == 'bcType':
                continue

            text, attrs = updates[subtag]
            element = parent.find(subtag)

            if element is None:
                element = ElementTree.SubElement(parent, subtag)

            set_xml_elem(element, text, attrs)


def append_xml_segment(root, segment, append_str):
    # Find the <segment> section
    controls_section = root.find(segment)

    if controls_section is not None:
        # Parse the controls_bodies string into XML elements
        controls_elements = ElementTree.fromstring(f"<wrapper>{append_str}</wrapper>")
        
        # Append each child of the parsed controls_bodies to the <bodies> section
        for child in controls_elements:
            controls_section.append(child)

    return controls_section

"""Provider-independent response tolerance without weakening required types."""

import unittest

from test_index import MENU, shared


class MenuValidationTests(unittest.TestCase):
    def test_empty_sections_are_preserved_even_without_populated_sections(self):
        empty = {"name": "Monday-Friday 3-6", "items": []}
        for sections in ([empty], [empty, *MENU["sections"], empty]):
            for provider in ("OpenAI", "Anthropic"):
                with self.subTest(sections=sections, provider=provider):
                    result = shared.validate_menu({**MENU, "sections": sections}, provider_name=provider)
                    self.assertEqual(result["sections"], sections)

    def test_blank_text_is_normalized_without_discarding_sections(self):
        sections = [
            {"name": "  ", "items": ["  Coffee. $4.  ", "", " \n "]},
            {"name": " Hours ", "items": ["", " \t "]},
        ]
        result = shared.validate_menu({**MENU, "restaurantName": " ", "sections": sections},
                                      provider_name="OpenAI")
        self.assertEqual(result, {"isDummy": False, "restaurantName": "Restaurant", "sections": [
            {"name": "", "items": ["Coffee. $4."]}, {"name": "Hours", "items": []},
        ]})
        self.assertEqual(sections[0]["name"], "  ")
        self.assertEqual(len(sections[0]["items"]), 3)

    def test_extra_metadata_is_ignored_not_forwarded(self):
        data = {**MENU, "extra": True, "isDummy": True, "sections": [
            {"name": "Drinks", "items": [], "source": "provider-only"},
        ]}
        self.assertEqual(shared.validate_menu(data, provider_name="OpenAI"), {
            "isDummy": False, "restaurantName": MENU["restaurantName"],
            "sections": [{"name": "Drinks", "items": []}],
        })

    def test_required_fields_and_types_remain_enforced(self):
        cases = [None, [], {}, {**MENU, "status": "unknown"}, {**MENU, "restaurantName": None},
                 {**MENU, "sections": []}, {**MENU, "sections": {}}]
        for key in MENU:
            cases.append({name: value for name, value in MENU.items() if name != key})
        for section in (None, {}, {"name": "Drinks"}, {"items": []},
                        {"name": None, "items": []}, {"name": 1, "items": []},
                        {"name": "Drinks", "items": None}, {"name": "Drinks", "items": "Coffee"},
                        {"name": "Drinks", "items": ["Coffee", None]},
                        {"name": "Drinks", "items": [1]},
                        {"name": "Drinks", "items": [{"name": "Coffee"}]}):
            cases.append({**MENU, "sections": [section]})
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(shared.ApiError) as caught:
                    shared.validate_menu(data, provider_name="OpenAI")
                self.assertEqual(caught.exception.status, 502)
                self.assertEqual(caught.exception.code, "invalid_openai_response")

    def test_explicit_failure_status_is_not_overridden_by_empty_sections(self):
        for status, code in (("no_menu", "no_menu_found"), ("illegible_text", "illegible_text")):
            with self.subTest(status=status):
                with self.assertRaises(shared.ApiError) as caught:
                    shared.validate_menu({**MENU, "status": status, "sections": [
                        {"name": "Hours", "items": []},
                    ]}, provider_name="OpenAI")
                self.assertEqual(caught.exception.status, 422)
                self.assertEqual(caught.exception.code, code)
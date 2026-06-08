from services.example_bills import example_bill_by_url, load_example_bills, normalize_example_url


def test_load_example_bills_has_expected_count():
    bills = load_example_bills()
    assert len(bills) == 14


def test_example_bill_lookup_normalizes_url():
    url = "https://moc.gov.gh/wp-content/uploads/2023/03/Data-Harmonisation-Bill.pdf/"
    bill = example_bill_by_url(url)
    assert bill is not None
    assert bill.id == "data-harmonisation-bill"


def test_normalize_example_url_lowercases_host_and_scheme():
    normalized = normalize_example_url("HTTPS://MOC.GOV.GH/wp-content/uploads/2023/03/Data-Harmonisation-Bill.pdf")
    assert normalized == "https://moc.gov.gh/wp-content/uploads/2023/03/Data-Harmonisation-Bill.pdf"

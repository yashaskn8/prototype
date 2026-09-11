from django.core.exceptions import ValidationError
from django.db.models.signals import m2m_changed, post_save
from django.dispatch import receiver

from .models import Project, ServiceCall
from .services.indexing import index_service_resolution


@receiver(post_save, sender=ServiceCall)
def keep_resolution_index_in_sync(sender, instance, raw=False, **kwargs):
    if raw:
        return
    index_service_resolution(instance)


@receiver(m2m_changed, sender=Project.assets.through)
def enforce_project_asset_scope(sender, instance, action, pk_set, **kwargs):
    """Prevent a project from linking assets from another tenant/customer."""
    if action != "pre_add" or not pk_set:
        return
    from .models import Asset

    invalid = Asset.objects.filter(pk__in=pk_set).exclude(tenant_id=instance.tenant_id)
    if invalid.exists():
        raise ValidationError("Project assets must belong to the same tenant as the project.")
    if instance.customer_id:
        wrong_customer = Asset.objects.filter(pk__in=pk_set).exclude(customer_id=instance.customer_id)
        if wrong_customer.exists():
            raise ValidationError("Project assets must belong to the project customer.")

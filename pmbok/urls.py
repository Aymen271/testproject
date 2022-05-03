from django.urls import path
from . import views
from django.urls import include, re_path

urlpatterns = [
    path('',views.index, name='index'),
    path('pmbokSection',views.pmbokSection,name='pmboksection'),
#    path('details/<section>/', views.displaySection, name='displaySection'),
#    url(r'^pdf', views.pdf, name='pdf'),
	path('viewpdf/<section>/', views.pdf_view, name='viewpdf'),
#	path('formUser',views.formUser,name='formUser'),
#	path('index',views.index,name='index')
    path('add', views.addTodo, name='add'),




]